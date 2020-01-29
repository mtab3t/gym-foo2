from gym.envs.registration import register

register(
    id='foo2-v0',
    entry_point='gym_foo2.envs:Foo2Env',
)