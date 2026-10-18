"""Documentation-only monetary notation and reversible formula checks."""
import re

RENAMES = (
    (r"\Pi^{cash}", r"C^{cash}"),
    (r"\Pi^{book}", r"C^{book}"),
    (r"G^{DR}", r"C^{DR,value}"),
    (r"K_i^{bill}", r"C_i^{bill}"),
    (r"P_i", r"C_i^{energy}"),
    (r"I_i", r"C_i^{DR}"),
    (r"A_i", r"C_i^{comp}"),
    (r"L_U", r"C^{UC,service}"),
    (r"C^r", r"C^{DR,orig}"),
)

CASH_FORMULA = r"""\begin{aligned}
C^{PG}&=wp^{PG,exec}\Delta t,\\
C^{DER,op}&=k_R(p^{DER}-p^{spill})\Delta t,\\
C^{cash}&=\sum_i C_i^{bill}-C^{PG}-C^{DER,op}-C^{ext}.
\end{aligned}\tag{24}"""
BOOK_FORMULA = r"""\begin{aligned}
C^{ESS}&=k_E(p^{ch,DER}+p^{ch,PG}+p^{dis,exec})\Delta t,\\
C^{spill}&=k_a p^{spill}\Delta t,\\
C^{book}&=C^{cash}-C^{ESS}-C^{spill}.
\end{aligned}\tag{25}"""
LEGACY_CASH = r"\Pi^{cash}=\sum_i K_i^{bill}-wp^{PG,exec}\Delta t-k_R(p^{DER}-p^{spill})\Delta t-C^{ext}.\tag{24}"
LEGACY_BOOK = r"\Pi^{book}=\Pi^{cash}-k_E(p^{ch,DER}+p^{ch,PG}+p^{dis,exec})\Delta t-k_ap^{spill}\Delta t.\tag{25}"


def rename_math(tex):
    for old, new in RENAMES:
        tex = tex.replace(old, new)
    return re.sub(r"C_i(?!\^|[A-Za-z0-9])", lambda _: r"C_i^{total}", tex)


def restore_symbols(tex):
    tex = tex.replace(r"C_i^{total}", "C_i")
    for old, new in reversed(RENAMES):
        tex = tex.replace(new, old)
    return tex


def compact(tex):
    return re.sub(r"\s+", "", tex)


def restore_formula(tex):
    # Strictly verify the new named components before eliminating their aliases.
    if r"\tag{24}" in tex and r"C^{PG}" in tex:
        assert compact(tex) == compact(CASH_FORMULA), "Cash definitions changed"
        return LEGACY_CASH
    if r"\tag{25}" in tex and r"C^{ESS}" in tex:
        assert compact(tex) == compact(BOOK_FORMULA), "Book-cost definitions changed"
        return LEGACY_BOOK
    return restore_symbols(tex)
