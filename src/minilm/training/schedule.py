import math


def get_lr_scheduler(
    start_factor: float,
    end_factor,
    warmup_steps: int,
    constant_steps: int,
    decay_steps: int,
):

    def lr_factor(step: int) -> float:
        # warmup
        if step < warmup_steps:
            progress = step / warmup_steps
            return start_factor + progress * (1.0 - start_factor)

        # constant
        if step < warmup_steps + constant_steps:
            return 1.0

        # Cosine decay
        schedule_end = warmup_steps + constant_steps + decay_steps

        if step < schedule_end:
            cosine_step = step - warmup_steps - constant_steps
            progress = cosine_step / decay_steps

            return end_factor + 0.5 * (1.0 - end_factor) * (
                1.0 + math.cos(math.pi * progress)
            )

        return end_factor

    return lr_factor
