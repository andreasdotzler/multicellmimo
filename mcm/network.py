import numpy as np


class Transmitter:
    def __init__(self, users_per_mode, As_per_mode):
        self.users_per_mode : dict[str, list[int]] = users_per_mode
        self.As_per_mode: dict[str, np.array] = As_per_mode
        self.modes = list(As_per_mode.keys())
        self.users = []
        for users in users_per_mode.values():
            self.users += users
        self.users = list(set(self.users))


    def wsr(self, weights, mode):
        t_weights = weights[self.users_per_mode[mode]]
        A = self.As_per_mode[mode]
        max_i = np.argmax(t_weights @ A)
        rates = A[:, max_i]
        return t_weights @ rates, rates


    #def util_fixed_fractions(self, fractions, util, q_min, q_max):
    #    return timesharing_fixed_fractions(util, fractions, self.users_per_mode, self.As_per_mode, q_min, q_max)

class Network:
    def __init__(self, users_per_mode_and_transmitter, As):
        self.users_per_mode_and_transmitter = users_per_mode_and_transmitter
        self.As = As
        self.transmitters = {}
        At = {}
        users_t = {}
        for mode in As:
            for transmitter, users in self.users_per_mode_and_transmitter[mode].items():
                if transmitter not in At:
                    At[transmitter] = {}
                At[transmitter][mode] = As[mode][transmitter]
                if transmitter not in users_t:
                    users_t[transmitter] = {}
                users_t[transmitter][mode] = users
        for transmitter in At:
            self.transmitters[transmitter] = Transmitter(users_t[transmitter], At[transmitter])
        users = []
        for t in self.transmitters.values():
            users += t.users
        self.users = list(set(users))    
        self.modes = list(As.keys())    


    #def wsr(self, weights):
    #    value, rates, _, _ = timesharing_network(weighted_sum_rate(weights), self.users_per_mode_and_transmitter, self.As)
    #    return value, rates


    def wsr_per_mode(self, weights):
        max_value = -np.Inf
        max_rates = None
        _, A_m = self.wsr_per_mode_and_transmitter(weights)
        for mode in self.As:
            mode_rates = np.zeros(len(weights))
            mode_value = 0
            for transmitter, users in self.users_per_mode_and_transmitter[mode].items():
                rates = A_m[mode][transmitter]
                value = weights[users] @ rates
                mode_rates[users] += rates
                mode_value += value
            if mode_value > max_value:
                max_value = mode_value
                max_rates = mode_rates
        assert len(weights) == len(max_rates)
        return max_value, max_rates

    def wsr_per_mode_and_transmitter(self, weights):
        values = {}
        A_max = {}
        # for mode in self.As:
        #     A_max[mode] = {}
        #     values[mode] = {}
        #     for transmitter, users in self.users_per_mode_and_transmitter[mode].items():
        #         t_weights = weights[users]
        #         # value, rates =  _, _ = timesharing_network(weighted_sum_rate(weights), {mode: self.users_per_mode_and_transmitter[mode]},
        #         #                                     {mode: self.As[mode]})
        #         #value, rates, alpha, lambda_phy = time_sharing(weighted_sum_rate(t_weights), self.As[mode][transmitter])
        #         A = self.As[mode][transmitter]
        #         max_i = np.argmax(t_weights @ A)
        #         rates = A[:, max_i]
        #         #assert t_weights @ rates == pytest.approx(value, rel=1e3, abs=1e-2)
        #         A_max[mode][transmitter] = rates
        #         values[mode][transmitter] = t_weights @ rates
        for transmitter_id, transmitter in self.transmitters.items():
            for mode in transmitter.As_per_mode:
                val, rates = transmitter.wsr(weights, mode)
                if mode not in values:
                    values[mode] = {}
                values[mode][transmitter_id] = val
                if mode not in A_max:
                    A_max[mode] = {}
                A_max[mode][transmitter_id] = rates
                
        return values, A_max

    def util_fixed_fractions(self, fractions, util, q_min, q_max):
        F = 0
        F_t_s = {}
        r = np.zeros(len(self.users))
        alphas = {}
        d_f = {}
        for transmitter_id, t in self.transmitters.items():
            F_t, r_t, alpha_t, [lambdas, w_min, w_max, d_f_t_m, d_c_m] = t.util_fixed_fractions(
                fractions, util, q_min[t.users], q_max[t.users])
            for mode, a in alpha_t.items():
                if mode not in alphas:
                    alphas[mode] = {}
                alphas[mode][transmitter_id] = a
            #for mode, d in d_f_t_m.items():
            #    if mode not in d_f:
            #        d_f[mode] = {}
            d_f[transmitter_id] = d_f_t_m
            F += F_t
            F_t_s[transmitter_id] = F_t
            for user, rate in r_t.items():
                r[user] += rate
        return F, r, alphas, d_f, F_t_s       
      
