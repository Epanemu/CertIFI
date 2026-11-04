def import_grb():
    """Import gurobipy lazily, with a helpful message if it is missing."""
    try:
        from gurobipy import GRB
    except ImportError as e:
        raise ImportError(
            "Training fair classifiers requires Gurobi (lazy-constraint callbacks). "
            "Install it with `pip install certifi-learn[gurobi]` and make sure a license is available."
        ) from e
    return GRB
