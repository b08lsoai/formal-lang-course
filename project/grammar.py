from pyformlang.cfg import CFG, Production, Terminal, Variable


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Weak Chomsky normal form: ``A -> B C``, ``A -> a`` and ``A -> epsilon``."""
    cfg = _lift_terminals(cfg)
    cfg = _binarize(cfg)
    cfg = cfg.eliminate_unit_productions()
    cfg = cfg.remove_useless_symbols()
    # pyformlang's unit elimination re-adds every production once per variable
    # (its unit pairs include the reflexive ones), so drop the copies.
    return CFG(start_symbol=cfg.start_symbol, productions=set(cfg.productions))


def _fresh_variable(prefix: str, used: set[Variable]) -> Variable:
    """First ``<prefix>_<n>`` that does not clash with the existing variables."""
    index = 0
    while True:
        variable = Variable(f"{prefix}_{index}")
        index += 1
        if variable not in used:
            used.add(variable)
            return variable


def _lift_terminals(cfg: CFG) -> CFG:
    """Lift terminals out of bodies longer than one symbol into fresh variables."""
    used = set(cfg.variables)
    preterminals: dict[Terminal, Variable] = {}
    productions: set[Production] = set()

    for production in cfg.productions:
        body = list(production.body)
        if len(body) > 1:
            for position, symbol in enumerate(body):
                if isinstance(symbol, Terminal):
                    if symbol not in preterminals:
                        preterminals[symbol] = _fresh_variable("T", used)
                        productions.add(Production(preterminals[symbol], [symbol]))
                    body[position] = preterminals[symbol]
        productions.add(Production(production.head, body))

    return CFG(start_symbol=cfg.start_symbol, productions=productions)


def _binarize(cfg: CFG) -> CFG:
    """Split every body longer than two symbols into a chain of binary rules."""
    used = set(cfg.variables)
    productions: set[Production] = set()

    for production in cfg.productions:
        body = list(production.body)
        while len(body) > 2:
            right = body.pop()
            left = body.pop()
            intermediate = _fresh_variable("B", used)
            productions.add(Production(intermediate, [left, right]))
            body.append(intermediate)
        productions.add(Production(production.head, body))

    return CFG(start_symbol=cfg.start_symbol, productions=productions)


class WCNFGrammar:
    """A CFG converted to weak Chomsky normal form and indexed by rule body.

    ``epsilon_productions`` contains heads of ``A -> epsilon`` rules.
    ``terminal_productions[a]`` contains heads of ``A -> a`` rules.
    ``binary_productions[B, C]`` contains heads of ``A -> B C`` rules.
    """

    def __init__(self, cfg: CFG) -> None:
        cfg = cfg_to_weak_normal_form(cfg)
        self.start_symbol = cfg.start_symbol
        self.variables = cfg.variables
        self.epsilon_productions: set[Variable] = set()
        self.terminal_productions: dict[str, set[Variable]] = {}
        self.binary_productions: dict[tuple[Variable, Variable], set[Variable]] = {}

        for production in cfg.productions:
            head, body = production.head, production.body
            if len(body) == 0:
                self.epsilon_productions.add(head)
            elif len(body) == 1:
                self.terminal_productions.setdefault(body[0].value, set()).add(head)
            else:
                self.binary_productions.setdefault((body[0], body[1]), set()).add(head)
