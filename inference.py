import torch
import torchode as to
from utils import lerp
from utils import get_obj_from_str


def t(n, device):
    return torch.linspace(0, 1, 2, device=device)[None, :].repeat(n, 1)


def unproject(x):
    return ((x + 1.0) / 2.0).clip(0.0, 1.0)


@torch.no_grad()
def sample(model, x0, t_eval, solver):
    c, h, w = x0.shape[1:]

    def flattened_model(t, y):
        y = y.reshape(-1, c, h, w)
        return model(y, t).flatten(start_dim=1)

    term = to.ODETerm(f=flattened_model)
    step_method = get_obj_from_str(solver.name)(term)
    if solver.dt0 is not None:
        step_size_controller = to.FixedStepController()
        dt0 = torch.full((x0.shape[0],), solver.dt0).to(x0)
    else:
        step_size_controller = to.IntegralController(
            atol=solver.atol, rtol=solver.rtol, term=term
        )
        dt0 = None
    adjoint = to.AutoDiffAdjoint(step_method, step_size_controller)
    problem = to.InitialValueProblem(y0=x0.flatten(start_dim=1), t_eval=t_eval)

    sol = adjoint.solve(problem, dt0=dt0)
    return sol.stats, sol.ys.reshape(*sol.ys.shape[:2], c, h, w)


def joint_ode(model, start, end):
    """
    return ODE model used to solve (t, alpha) from start to end, as a straight line in terms of k (from 0 to 1)
    """
    t_0, alpha_0 = start
    t_1, alpha_1 = end

    def _func(x, k):
        t = lerp(t_0, t_1, k)
        alpha = lerp(alpha_0, alpha_1, k)

        v, d = torch.chunk(model(x, t, alpha), 2, dim=1)

        return v * (t_1 - t_0) + d * (alpha_1 - alpha_0)

    return _func


def condiff(model, cond):
    def _func(x, alpha):
        return model(torch.cat([x, cond], dim=1), alpha)

    return _func
