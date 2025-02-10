import torch
from diffuser.models.helpers import extract, apply_conditioning

@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    """
    Flowmatching 방식에 맞춘 guided sampling 함수입니다.
    
    기존 diffusion에서는 posterior_log_variance_clipped를 이용하여 gradient를 스케일링하였으나,
    flowmatching에서는 해당 버퍼가 존재하지 않으므로, scale_grad_by_std 옵션은 현재 아무런 효과를 내지 않습니다.
    필요하다면 여기서 다른 스케일링 전략을 적용할 수 있습니다.
    
    인자:
      model: flowmatching 방식의 diffusion 모델
      x: 현재 샘플 텐서
      cond: 조건 정보
      t: 현재 timestep (각 배치마다 동일 혹은 다르게)
      guide: gradient 정보를 제공하는 객체 (guide.gradients 메서드가 있어야 함)
      scale: gradient 업데이트 스케일 (기본값: 0.001)
      t_stopgrad: t 미만에서는 가이드 gradient를 무시 (기본값: 0)
      n_guide_steps: 가이드 업데이트를 몇 번 수행할지 (기본값: 1)
      scale_grad_by_std: 기존 diffusion에서는 posterior variance로 스케일링하는 옵션 (flowmatching에서는 무시됨)
      
    반환:
      x_updated: 가이드 업데이트와 p_mean_variance를 거친 새로운 샘플
      y: 마지막 guide.gradients의 y 값
    """
    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)
        
        if scale_grad_by_std:
            # flowmatching에서는 posterior_log_variance가 존재하지 않으므로, 이 부분은 생략합니다.
            # 필요하다면 다른 스케일링 값을 적용할 수 있습니다.
            pass

        # t < t_stopgrad 인 경우 gradient 업데이트를 중단합니다.
        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    # flowmatching에서는 p_mean_variance가 결정론적으로 x를 업데이트합니다.
    x_updated = model.p_mean_variance(x=x, cond=cond, t=t)

    return x_updated, y