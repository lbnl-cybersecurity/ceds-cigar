from pycigar.envs import Env
import yaml
import time

class FooEnv(Env):
    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'),
                   shape=(5,), dtype=np.float64)

    @property
    def action_space(self):
        return Box(low=0.5, high=1.5, shape=(5,), dtype=np.float64)

    def step(self, rl_actions=None, randomize_rl_update=None):
        """See parent class.
        """

        for _ in range(self.sim_params['env_config']["sims_per_step"]):
            self.env_time += 1

            # perform action update for PV inverter device
            if len(self.k.device.get_adaptive_device_ids()) > 0:
                control_setting = []
                for device_id in self.k.device.get_adaptive_device_ids():
                    action = self.k.device.get_controller(device_id).get_action(self)
                    control_setting.append(action)
                self.k.device.apply_control(self.k.device.get_adaptive_device_ids(), control_setting)

            # perform action update for PV inverter device
            if len(self.k.device.get_fixed_device_ids()) > 0:
                control_setting = []
                for device_id in self.k.device.get_fixed_device_ids():
                    action = self.k.device.get_controller(device_id).get_action(self)
                    control_setting.append(action)
                self.k.device.apply_control(self.k.device.get_fixed_device_ids(), control_setting)

            self.additional_command()

            if self.k.time <= self.k.t:
                self.k.update(reset=False)

                # check whether the simulator sucessfully solved the powerflow
                converged = self.k.simulation.check_converged()
                if not converged:
                    break

            if self.k.time >= self.k.t:
                break

        # the episode will be finished if it is not converged.
        done = not converged or (self.k.time == self.k.t)
        obs = self.get_state()
        infos = {}
        reward = self.compute_reward(rl_actions)

        return obs, reward, done, infos

    def get_state(self):
        return [0, 0, 0, 0, 0]

    def compute_reward(self, rl_actions, **kwargs):
        return 0


from pycigar.utils.input_parser import input_parser
misc_inputs_path = '../data/ieee37busdata/misc_inputs.csv'
dss_path = '../data/ieee37busdata/ieee37.dss'
load_solar_path = '../data/ieee37busdata/load_solar_data.csv'
breakpoints_path = '../data/ieee37busdata/breakpoints.csv'
sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path, benchmark=False, percentage_hack=0.45, adv=False)

env = FooEnv(sim_params)
total_start_time = time.time()
env.reset()
done = False
while not done:
    start_time = time.time()
    _, _, done, _ = env.step()
    print("----- {} seconds ----".format(time.time()-start_time))

print("----- total: {} seconds ----".format(time.time()-total_start_time))
print("----- mean: {} seconds ----".format((time.time()-total_start_time)/20))