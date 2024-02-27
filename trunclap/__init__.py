import traceback

try:
    from .truncated_laplace import TruncatedLaplaceError, TruncatedLaplace
except TruncatedLaplaceError:
    traceback.print_exc()
