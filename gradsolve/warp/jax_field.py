"""Translate a user ``f_jax(t, y, p)`` into a fused Warp ``@wp.func`` field.

A user writes their RHS once in JAX; ``jaxpr_to_warp_field`` walks
``jax.make_jaxpr(f_jax)(t, y, p)`` and emits a compiled ``@wp.func field(t, y: vecD,
p: vecP)`` whose value matches a hand-written field to float64 round-off. The generated
field then plugs into the same fused adaptive-Tsit5 / Rosenbrock23 kernels the built-ins
(Lorenz, VdP, Robertson, HIRES) already call, so a matching ``Problem`` routes to the
fused engines exactly as a built-in does.

NO ``from __future__ import annotations`` — same codegen-boundary reason as
``_warp_kernel.py``: Warp evaluates the generated ``@wp.func`` annotations
(``wp_scalar``, ``vecD``, ``vecP``) as LIVE objects, so they must be real objects in the
function's globals, not strings.

Mechanism. Every jaxpr value is represented as a flat list of scalar Python-source
strings (its components, row-major) plus its shape. Reshaping primitives (squeeze,
slice, concatenate, stack, broadcast_in_dim, reshape, expand_dims, convert_element_type) just
rearrange the component-string lists — done by running the corresponding numpy op on an
object array of the strings, so nothing is emitted. Arithmetic primitives (add, mul,
exp, ...) emit one ``_vN = <expr>`` scalar assignment line per output component. Loops
are unrolled; every intermediate is a scalar. Anything outside the supported subset
raises ``UnsupportedRHS`` naming the offending primitive (never a silent fallback).
"""
import itertools
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

_PRECISION_SCALARS = {wp.float64, wp.float32}


class UnsupportedRHS(Exception):
    """A user ``f_jax`` uses a jaxpr primitive outside the translatable subset.

    Carries ``primitive_name`` and a ``hint``; ``str`` names the primitive so
    ``fused="auto"`` can record a clear fallback reason.
    """

    def __init__(self, primitive_name, hint=""):
        self.primitive_name = primitive_name
        self.hint = hint
        msg = f"unsupported jaxpr primitive {primitive_name!r} in f_jax"
        if hint:
            msg += f": {hint}"
        super().__init__(msg)


# ---------------------------------------------------------------------------------
# Value = (shape, comps). comps is a flat row-major list of scalar Python-source
# strings; len(comps) == prod(shape) (a scalar is shape () with one comp).
# ---------------------------------------------------------------------------------

def _lit_str(val, wp_scalar_name):
    return f"{wp_scalar_name}({float(val)!r})"


def _is_literal(x):
    return type(x).__name__ == "Literal"


def _as_number(v):
    """Coerce a jaxpr literal (numpy/python scalar) to a plain Python number/bool."""
    return np.asarray(v).item()


def _is_num(c):
    """True for a folded numeric/bool component (not a source string)."""
    return isinstance(c, (int, float, bool)) and not isinstance(c, str)


# op -> source template ({a}, {b}); op -> numeric fold; unary op -> numeric fold.
_OP_TMPL = {
    "add": "({a} + {b})", "sub": "({a} - {b})", "mul": "({a} * {b})",
    "div": "({a} / {b})", "max": "wp.max({a}, {b})", "min": "wp.min({a}, {b})",
    "pow": "wp.pow({a}, {b})",
    "lt": "({a} < {b})", "le": "({a} <= {b})", "gt": "({a} > {b})",
    "ge": "({a} >= {b})", "eq": "({a} == {b})", "ne": "({a} != {b})",
    "neg": "(-{a})", "abs": "wp.abs({a})", "sign": "wp.sign({a})",
    "exp": "wp.exp({a})", "log": "wp.log({a})", "sin": "wp.sin({a})",
    "cos": "wp.cos({a})", "tan": "wp.tan({a})", "tanh": "wp.tanh({a})",
    "sqrt": "wp.sqrt({a})",
}
_NUM_FOLD = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
    "max": max, "min": min, "pow": lambda a, b: a ** b,
    "lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
}
_NUM_UNARY = {"neg": lambda x: -x, "abs": abs, "sign": lambda x: float(np.sign(x))}


class _Emitter:
    """Walks a jaxpr, emitting scalar assignment lines; builds the ``@wp.func`` source."""

    def __init__(self, wp_scalar_name):
        self.lines = []
        self.n = 0
        self.wp_scalar_name = wp_scalar_name

    def emit(self, expr):
        name = f"_v{self.n}"
        self.n += 1
        self.lines.append(f"{name} = {expr}")
        return name

    # -- value helpers ------------------------------------------------------------
    def _val_of(self, atom, env):
        if _is_literal(atom):
            # keep literals as live Python numbers so constant-only sub-expressions
            # (the jacfwd identity seed: iota/eq/convert) fold away instead of
            # emitting bool/int kernel code Warp cannot type.
            return ((), [_as_number(atom.val)])
        return env[id(atom)]

    def _src(self, c):
        """Component -> Warp source string (numbers become typed literals)."""
        if isinstance(c, str):
            return c
        return _lit_str(float(c), self.wp_scalar_name)

    def _as_obj_array(self, val):
        shape, comps = val
        return np.array(comps, dtype=object).reshape(shape)

    def _from_obj_array(self, arr):
        return (tuple(arr.shape), list(arr.reshape(-1)))

    # -- broadcasting for elementwise ops ----------------------------------------
    def _broadcast_pair(self, a, b):
        (sa, ca), (sb, cb) = a, b
        na, nb = len(ca), len(cb)
        if na == nb:
            shape = sa if len(sa) >= len(sb) else sb
            return shape, ca, cb
        if na == 1:
            return sb, ca * nb, cb
        if nb == 1:
            return sa, ca, cb * na
        raise UnsupportedRHS(
            "broadcast",
            f"cannot broadcast shapes {sa} and {sb} in the fused-field subset",
        )

    # -- primitive handlers -------------------------------------------------------
    def _binop(self, a, b, op):
        shape, ca, cb = self._broadcast_pair(a, b)
        return (shape, [self._scalar_binop(op, xa, xb) for xa, xb in zip(ca, cb)])

    def _scalar_binop(self, op, xa, xb):
        na, nb = _is_num(xa), _is_num(xb)
        if na and nb:
            return _NUM_FOLD[op](xa, xb)
        # zero/one absorption: keeps generated Jacobians sparse and their structural
        # zeros exactly 0.0 (jacfwd seeds the identity, so tangents multiply by 0/1).
        if op == "mul":
            if (na and xa == 0) or (nb and xb == 0):
                return 0.0
            if na and xa == 1:
                return xb
            if nb and xb == 1:
                return xa
        elif op == "add":
            if na and xa == 0:
                return xb
            if nb and xb == 0:
                return xa
        elif op == "sub":
            if nb and xb == 0:
                return xa
        return self.emit(_OP_TMPL[op].format(a=self._src(xa), b=self._src(xb)))

    def _unary(self, a, op):
        shape, comps = a
        out = []
        for c in comps:
            if _is_num(c) and op in _NUM_UNARY:
                out.append(_NUM_UNARY[op](c))
            else:
                out.append(self.emit(_OP_TMPL[op].format(a=self._src(c))))
        return (shape, out)

    def handle(self, prim, invals, params):
        h = _HANDLERS.get(prim)
        if h is None:
            raise UnsupportedRHS(prim, "not in the fused-field primitive subset")
        return h(self, invals, params)


# Each handler: (emitter, invals, params) -> Value. invals are resolved Values.
def _make_binop(op):
    def h(e, iv, p):
        return e._binop(iv[0], iv[1], op)
    return h


def _make_unary(op):
    def h(e, iv, p):
        return e._unary(iv[0], op)
    return h


def _h_integer_pow(e, iv, p):
    n = int(p["y"])
    shape, comps = iv[0]
    out = []
    for c in comps:
        if _is_num(c):
            out.append(c ** n)
        else:
            out.append(e.emit("wp.pow(%s, %s(%r))" % (
                e._src(c), e.wp_scalar_name, float(n))))
    return (shape, out)


def _h_convert(e, iv, p):
    # all one precision inside the kernel; element-type conversion is a no-op
    # (bool from an eq fold flows through and becomes a 0.0/1.0 literal at _src).
    return iv[0]


def _h_iota(e, iv, p):
    # constant index grid; folds away with the eq that seeds the jacfwd identity.
    shape = tuple(p["shape"])
    grid = np.indices(shape)[p["dimension"]]
    return (shape, [int(v) for v in grid.reshape(-1)])


def _h_transpose(e, iv, p):
    arr = e._as_obj_array(iv[0])
    return e._from_obj_array(np.transpose(arr, tuple(p["permutation"])))


def _h_split(e, iv, p):
    sizes = [int(s) for s in p["sizes"]]
    if len(sizes) != 1:
        raise UnsupportedRHS("split", "multi-output split not supported")
    return e._from_obj_array(e._as_obj_array(iv[0]))


def _h_squeeze(e, iv, p):
    arr = e._as_obj_array(iv[0])
    return e._from_obj_array(np.squeeze(arr, axis=tuple(p["dimensions"])))


def _h_expand_dims(e, iv, p):
    arr = e._as_obj_array(iv[0])
    return e._from_obj_array(np.expand_dims(arr, tuple(p["dimensions"])))


def _h_reshape(e, iv, p):
    arr = e._as_obj_array(iv[0])
    return e._from_obj_array(arr.reshape(tuple(p["new_sizes"])))


def _h_broadcast_in_dim(e, iv, p):
    arr = e._as_obj_array(iv[0])
    out_shape = tuple(p["shape"])
    bdims = tuple(p["broadcast_dimensions"])
    inter = [1] * len(out_shape)
    for i, d in enumerate(bdims):
        inter[d] = arr.shape[i]
    tmp = arr.reshape(inter)
    return e._from_obj_array(np.broadcast_to(tmp, out_shape))


def _h_slice(e, iv, p):
    arr = e._as_obj_array(iv[0])
    starts = p["start_indices"]
    limits = p["limit_indices"]
    strides = p["strides"] or [1] * len(starts)
    idx = tuple(slice(s, lim, st) for s, lim, st in zip(starts, limits, strides))
    return e._from_obj_array(arr[idx])


def _h_concatenate(e, iv, p):
    arrs = [e._as_obj_array(v) for v in iv]
    return e._from_obj_array(np.concatenate(arrs, axis=p["dimension"]))


def _h_stack(e, iv, p):
    # jax >= 0.10.2 lowers jnp.stack to its own single-output `stack` primitive
    # (jax <= 0.10.1 lowered it to expand_dims + concatenate, both already handled).
    # stack_p.bind(*arrays, axis=axis): insert a new axis at `axis`, concatenate —
    # exactly np.stack over the component-string object arrays. `axis` is already
    # canonicalized non-negative by lax.stack.
    arrs = [e._as_obj_array(v) for v in iv]
    return e._from_obj_array(np.stack(arrs, axis=p["axis"]))


def _h_pjit(e, iv, p):
    # inline the inner jaxpr (pjit / jit / closed_call wrap a sub-computation)
    inner = p.get("jaxpr")
    if inner is None:
        raise UnsupportedRHS("pjit", "call primitive without an inlinable jaxpr")
    sub = getattr(inner, "jaxpr", inner)
    return e.walk_subjaxpr(sub, iv)


_HANDLERS = {
    "integer_pow": _h_integer_pow,
    "convert_element_type": _h_convert,
    "iota": _h_iota,
    "transpose": _h_transpose,
    "split": _h_split,
    "squeeze": _h_squeeze,
    "expand_dims": _h_expand_dims,
    "reshape": _h_reshape,
    "broadcast_in_dim": _h_broadcast_in_dim,
    "slice": _h_slice,
    "concatenate": _h_concatenate,
    "stack": _h_stack,
    "pjit": _h_pjit,
    "jit": _h_pjit,
    "closed_call": _h_pjit,
}
# add_any is jvp's elementwise "add tangents"; same emission as add.
for _op in ("add", "sub", "mul", "div", "max", "min", "pow",
            "lt", "le", "gt", "ge", "eq", "ne"):
    _HANDLERS[_op] = _make_binop(_op)
_HANDLERS["add_any"] = _make_binop("add")
for _op in ("neg", "abs", "sign", "exp", "log", "sin", "cos", "tan", "tanh", "sqrt"):
    _HANDLERS[_op] = _make_unary(_op)


def _walk(emitter, jaxpr, invals):
    """Walk ``jaxpr.eqns`` binding ``invals`` to invars; return the outvar Values."""
    env = {}
    for var, val in zip(jaxpr.invars, invals):
        env[id(var)] = val
    for eqn in jaxpr.eqns:
        resolved = [emitter._val_of(a, env) for a in eqn.invars]
        out = emitter.handle(eqn.primitive.name, resolved, eqn.params)
        outvars = eqn.outvars
        if len(outvars) == 1:
            env[id(outvars[0])] = out
        else:
            # multi-output primitives are not in the subset; guard defensively
            raise UnsupportedRHS(
                eqn.primitive.name, "multiple-output primitive not supported")
    return [emitter._val_of(a, env) for a in jaxpr.outvars]


# give the emitter a bound sub-walker for pjit inlining
def _walk_subjaxpr(self, jaxpr, invals):
    return _walk(self, jaxpr, invals)[0]


_Emitter.walk_subjaxpr = _walk_subjaxpr


_FIELD_CACHE = {}


def _cache_key(f_jax, dim, n_params, wp_scalar):
    underlying = getattr(f_jax, "__func__", f_jax)
    bound = getattr(f_jax, "__self__", None)
    return (id(underlying), id(bound), int(dim), int(n_params), wp_scalar)


# Warp evaluates a @wp.func's SOURCE (via inspect.getsourcelines) at codegen time, so the
# function cannot come from a bare exec() string — it must live in a real importable file.
# We write each generated field to a throwaway .py in a per-process temp dir and exec its
# compiled code with the dynamic types (vecD/vecP/wp_scalar) injected as module globals.
_GEN_DIR = tempfile.mkdtemp(prefix="gradsolve_genfield_")
_GEN_COUNTER = itertools.count()


def _emit_func(base_name, body_lines, return_expr, sig, namespace):
    k = next(_GEN_COUNTER)
    fname = f"{base_name}_{k}"
    modname = f"gradsolve_genfield_{k}"
    src = "\n".join(
        ["@wp.func", f"def {fname}({sig}):"]
        + [f"    {ln}" for ln in body_lines]
        + [f"    return {return_expr}", ""]
    )
    path = os.path.join(_GEN_DIR, modname + ".py")
    with open(path, "w") as fh:
        fh.write(src)
    namespace["__name__"] = modname
    namespace["__file__"] = path
    code = compile(src, path, "exec")  # co_filename == path so getsourcelines finds it
    exec(code, namespace)
    return namespace[fname]


def jaxpr_to_warp_field(f_jax, dim, n_params, *, wp_scalar):
    """Translate ``f_jax(t, y, p)`` into ``(field @wp.func, vecD, vecP)``.

    ``field(t, y: vecD, p: vecP) -> vecD`` matches the hand-written field to float64
    round-off (same textual arithmetic order as the jaxpr, so the built-in fields agree
    to ~1e-15). ``vecD = wp.types.vector(length=dim, dtype=wp_scalar)``;
    ``vecP = wp.types.vector(length=max(n_params, 1), dtype=wp_scalar)``.

    Raises ``UnsupportedRHS`` (naming the primitive) for any RHS using a primitive
    outside the translatable subset.
    """
    if wp_scalar not in _PRECISION_SCALARS:
        raise ValueError(f"wp_scalar must be one of {_PRECISION_SCALARS}, got {wp_scalar!r}")
    key = _cache_key(f_jax, dim, n_params, wp_scalar)
    cached = _FIELD_CACHE.get(key)
    if cached is not None:
        return cached

    P = max(int(n_params), 1)
    vecD = wp.types.vector(length=int(dim), dtype=wp_scalar)
    vecP = wp.types.vector(length=P, dtype=wp_scalar)

    closed = jax.make_jaxpr(f_jax)(0.0, jnp.zeros(dim), jnp.zeros(int(n_params)))
    jaxpr = closed.jaxpr
    if closed.consts:
        # constants folded into the closure would need to be materialized; the RHS
        # subset assumes literals inline. None of the built-ins hit this.
        raise UnsupportedRHS(
            "const", "f_jax closes over array constants (not inlinable literals)")

    wp_scalar_name = "wp.float64" if wp_scalar is wp.float64 else "wp.float32"
    emitter = _Emitter(wp_scalar_name)
    # invars: t (scalar), y (vecD), p (vecP). Seed component strings from the args.
    t_val = ((), ["t"])
    y_val = ((int(dim),), [f"y[{i}]" for i in range(int(dim))])
    p_val = ((int(n_params),), [f"p[{i}]" for i in range(int(n_params))]) if int(n_params) > 0 \
        else ((), ["p[0]"])
    outvals = _walk(emitter, jaxpr, [t_val, y_val, p_val])
    out_shape, out_comps = outvals[0]
    if len(out_comps) != int(dim):
        raise UnsupportedRHS(
            "shape", f"f_jax returns {len(out_comps)} components, expected dim={dim}")

    namespace = {"wp": wp, "wp_scalar": wp_scalar, "vecD": vecD, "vecP": vecP}
    comps_src = ", ".join(emitter._src(c) for c in out_comps)
    field = _emit_func(
        "field",
        emitter.lines,
        f"vecD({comps_src})",
        "t: wp_scalar, y: vecD, p: vecP",
        namespace,
    )
    result = (field, vecD, vecP)
    _FIELD_CACHE[key] = result
    return result


_JAC_CACHE = {}


def jaxpr_to_warp_jacobian(f_jax, dim, n_params, *, wp_scalar):
    """Translate ``jacfwd(f_jax, argnums=1)`` into ``(jac @wp.func, vecD, matDD, vecP)``.

    ``jac(t, y: vecD, p: vecD) -> matDD`` is the analytic df/dy the stiff Rosenbrock23
    kernel calls; it matches the hand-written Jacobian to float64 round-off. The
    ``jacfwd`` jaxpr seeds an identity basis with ``iota``/``eq``/``convert_element_type``;
    those are constant and fold away in the translator, so the emitted kernel contains
    only the field's own arithmetic (no bool/int ops Warp cannot type).

    Param convention (stiff): the kernel hands a length-``dim`` ``vecD`` param, so the
    generated ``jac`` takes ``p: vecD`` and reads ``p[0..n_params-1]``; ``n_params`` must
    be ``<= dim`` (``UnsupportedRHS`` otherwise). ``matDD =
    wp.types.matrix(shape=(dim, dim), dtype=wp_scalar)``. The fourth return value is the
    param vector type (``vecD`` here) for signature uniformity with ``jaxpr_to_warp_field``.
    """
    if wp_scalar not in _PRECISION_SCALARS:
        raise ValueError(f"wp_scalar must be one of {_PRECISION_SCALARS}, got {wp_scalar!r}")
    if int(n_params) > int(dim):
        raise UnsupportedRHS(
            "jacobian",
            f"stiff generated field needs n_params <= dim; got n_params={n_params}, "
            f"dim={dim} (the stiff kernel hands a length-dim vecD param)",
        )
    key = _cache_key(f_jax, dim, n_params, wp_scalar)
    cached = _JAC_CACHE.get(key)
    if cached is not None:
        return cached

    D = int(dim)
    vecD = wp.types.vector(length=D, dtype=wp_scalar)
    matDD = wp.types.matrix(shape=(D, D), dtype=wp_scalar)

    jacfun = jax.jacfwd(f_jax, argnums=1)
    closed = jax.make_jaxpr(jacfun)(0.0, jnp.zeros(D), jnp.zeros(int(n_params)))
    jaxpr = closed.jaxpr
    if closed.consts:
        raise UnsupportedRHS(
            "const", "jacfwd(f_jax) closes over array constants (not inlinable literals)")

    wp_scalar_name = "wp.float64" if wp_scalar is wp.float64 else "wp.float32"
    emitter = _Emitter(wp_scalar_name)
    t_val = ((), ["t"])
    y_val = ((D,), [f"y[{i}]" for i in range(D)])
    p_val = ((int(n_params),), [f"p[{i}]" for i in range(int(n_params))]) \
        if int(n_params) > 0 else ((), ["p[0]"])
    outvals = _walk(emitter, jaxpr, [t_val, y_val, p_val])
    out_shape, out_comps = outvals[0]
    if len(out_comps) != D * D:
        raise UnsupportedRHS(
            "shape", f"jacfwd(f_jax) returns {len(out_comps)} entries, expected {D * D}")

    body = list(emitter.lines) + ["J = matDD()"]
    for i in range(D):
        for j in range(D):
            body.append(f"J[{i}, {j}] = {emitter._src(out_comps[i * D + j])}")

    namespace = {"wp": wp, "wp_scalar": wp_scalar, "vecD": vecD, "matDD": matDD}
    jac = _emit_func("jac", body, "J", "t: wp_scalar, y: vecD, p: vecD", namespace)
    result = (jac, vecD, matDD, vecD)
    _JAC_CACHE[key] = result
    return result


def register_jax_field(name, f_jax, dim, n_params, *, stiff=False):
    """Register a translated ``f_jax(t, y, p)`` into the fused-kernel registries.

    A ``Problem`` whose ``name`` matches ``name`` then routes to the fused engines
    exactly as a built-in (Lorenz/VdP/Robertson/HIRES) does. Idempotent — a ``name``
    already registered is left untouched (``register_field`` uses ``setdefault``).

    Non-stiff (``stiff=False``): builds the field via :func:`jaxpr_to_warp_field` and
    registers it into ``_warp_kernel._FIELD_REGISTRY`` with the ``vecp=True`` marker, so
    ``warp_ode._launch`` hands the kernel a length-``n_params`` ``vecP`` param array (the
    built-in scalar fields are unaffected).

    Stiff (``stiff=True``): builds the field AND the analytic Jacobian
    (:func:`jaxpr_to_warp_jacobian`) and registers the pair into
    ``warp_rosenbrock``'s registry. The stiff kernel hands a length-``dim`` ``vecD``
    param, so the generated field is emitted with ``p: vecD`` (``n_params`` passed to the
    field translator as ``dim``) and indexes the leading ``n_params`` entries; requires
    ``n_params <= dim`` (raised by the Jacobian translator otherwise).

    The translation runs lazily at kernel-factory time (inside the ``builder`` closure),
    so registration itself does no codegen — the field compiles when the fused kernel is
    first built for a given precision.
    """
    dim = int(dim)
    n_params = int(n_params)

    if stiff:
        from gradsolve.warp import warp_rosenbrock

        def builder(wp_scalar):
            # p: vecD (length dim) — pass n_params=dim to the field translator so its
            # vecP is the length-dim vector the stiff kernel's param array uses. f_jax
            # reads only p[0..n_params-1] (n_params <= dim), so the extra slots are unused.
            field, vecD, _vecP = jaxpr_to_warp_field(
                f_jax, dim, dim, wp_scalar=wp_scalar)
            jac, _vecDj, matDD, _vecPj = jaxpr_to_warp_jacobian(
                f_jax, dim, n_params, wp_scalar=wp_scalar)
            return field, jac, vecD, matDD

        warp_rosenbrock.register_field(name, builder, dim)
    else:
        from gradsolve.warp import _warp_kernel

        def builder(wp_scalar):
            field, vecD, _vecP = jaxpr_to_warp_field(
                f_jax, dim, n_params, wp_scalar=wp_scalar)
            return field, vecD

        _warp_kernel.register_field(name, builder, dim, vecp=True)
