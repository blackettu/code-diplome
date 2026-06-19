from __future__ import annotations

from dataclasses import asdict, dataclass, field

from seedling_rl.envs import SeedlingTrayEnv, TrayEnvConfig


@dataclass(frozen=True)
class BaselineEvaluationResult:
    policy: str
    episodes: int
    reward_mean: float
    successful_target_rate: float
    critical_error_rate: float
    review_rate: float
    actions_per_tray: float
    total_distance_mm: float | None = None
    mean_distance_error_mm: float | None = None
    details: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_baseline_policy(
    policy: str,
    config: TrayEnvConfig | dict[str, object] | None = None,
    episodes: int = 10,
) -> BaselineEvaluationResult:
    env = SeedlingTrayEnv(config)
    rewards: list[float] = []
    successes = 0
    targets = 0
    critical = 0
    reviews = 0
    actions = 0
    details: list[dict[str, object]] = []
    for episode in range(episodes):
        obs, info = env.reset(seed=env.config.seed + episode)
        done = False
        episode_reward = 0.0
        episode_actions = 0
        while not done:
            action = select_baseline_action(policy, env, obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward
            episode_actions += 1
            event = step_info.get("event")
            if event == "target_step":
                outcome = step_info["outcome"]
                targets += 1
                if outcome.get("success"):
                    successes += 1
                if outcome.get("crop_damage"):
                    critical += 1
            elif event == "review":
                reviews += int(step_info.get("reviewed_targets", 0))
            elif event in {"unsafe_target", "invalid_target_index", "invalid_action"}:
                critical += 1
            done = terminated or truncated
        rewards.append(episode_reward)
        actions += episode_actions
        details.append({"episode": episode, "reward": episode_reward, "actions": episode_actions})
    return BaselineEvaluationResult(
        policy=policy,
        episodes=episodes,
        reward_mean=sum(rewards) / len(rewards) if rewards else 0.0,
        successful_target_rate=successes / targets if targets else 0.0,
        critical_error_rate=critical / max(actions, 1),
        review_rate=reviews / max(actions, 1),
        actions_per_tray=actions / max(episodes, 1),
        details=details,
    )


def evaluate_baseline_suite(
    config: TrayEnvConfig | dict[str, object] | None = None,
    episodes: int = 10,
    policies: list[str] | None = None,
) -> list[BaselineEvaluationResult]:
    policies = policies or ["noop", "raster_scan", "nearest_neighbor", "route_planning", "risk_aware_rule"]
    return [evaluate_baseline_policy(policy, config, episodes) for policy in policies]


def select_baseline_action(policy: str, env: SeedlingTrayEnv, obs: dict[str, object]) -> int:
    mask = obs["action_mask"]
    valid_indices = [index for index, value in enumerate(mask) if int(value) == 1]
    target_indices = [index for index in valid_indices if index < env.config.max_targets]
    if policy == "noop":
        return env.review_action if env.review_action in valid_indices else env.stop_action
    if policy in {"raster_scan", "risk_aware_rule"}:
        return target_indices[0] if target_indices else env.stop_action
    if policy in {"nearest_neighbor", "route_planning"}:
        if not target_indices:
            return env.stop_action
        robot = obs["robot_state"]
        features = obs["target_features"]
        return min(
            target_indices,
            key=lambda index: (features[index][0] - robot[0]) ** 2 + (features[index][1] - robot[1]) ** 2,
        )
    raise ValueError(f"Unknown baseline policy: {policy}")


_select_action = select_baseline_action
