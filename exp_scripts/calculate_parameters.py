def new_dem(i, p, f, s):
    return abs(((i + (2 * p) - f) // s) + 1)


def calculate_parameters(L_1, L):
    L_type = L['type']
    if L_type == 'CONV' or L_type == 'POOL':
        f = L['filter_size']
        p = L['padding']
        s = L['stride']
        L = {'filter_dem': [f, f, L_1['c']],
             'h': new_dem(L_1['h'], p, f, s),
             'w': new_dem(L_1['w'], p, f, s),
             'c': L['number_of_filters'] if L_type == 'CONV' else L_1['c']
             }

        L['activation_dem'] = [L['h'], L['w'], L['c']]
        L['activation_size'] = L['h'] * L['w'] * L['c']
        if L_type == 'CONV':
            L['weights_dem'] = f * f * L_1['c'] * L['c']
            L['bias_dem'] = L['c']  # (1, 1, 1, L['c'])
        else:
            L['weights_dem'] = 0
            L['bias_dem'] = 0
    elif L_type == 'FC':
        pass
    return L


def print_info(ll):
    l_1 = None
    total_parameters = 0
    for l in range(len(ll)):
        if ll[l]['type'] == 'INPUT':
            l_1 = ll[l]
            continue

        l_1 = calculate_parameters(l_1, ll[l])
        total_parameters += (l_1['weights_dem'] + l_1['bias_dem'])
        # print(l, l_1, l_1['weights_dem'] + l_1['bias_dem'])
    print(total_parameters)


SimpleCNN = [
    {'type': 'INPUT', 'h': 32, 'w': 32, 'c': 3},
    {'type': 'CONV', 'filter_size': 3, 'padding': 1, 'stride': 1, 'number_of_filters': 32},
    {'type': 'CONV', 'filter_size': 3, 'padding': 0, 'stride': 1, 'number_of_filters': 32},
    {'type': 'POOL', 'filter_size': 2, 'padding': 0, 'stride': 2, 'number_of_filters': 0},
    {'type': 'CONV', 'filter_size': 3, 'padding': 1, 'stride': 1, 'number_of_filters': 64},
    {'type': 'CONV', 'filter_size': 1, 'padding': 0, 'stride': 1, 'number_of_filters': 64}
]

CNN = [
    {'type': 'INPUT', 'h': 32, 'w': 32, 'c': 3},
    {'type': 'CONV', 'filter_size': 3, 'padding': 1, 'stride': 1, 'number_of_filters': 32},
    {'type': 'CONV', 'filter_size': 3, 'padding': 0, 'stride': 1, 'number_of_filters': 32},
    {'type': 'POOL', 'filter_size': 2, 'padding': 0, 'stride': 2, 'number_of_filters': 0},
    {'type': 'CONV', 'filter_size': 3, 'padding': 1, 'stride': 1, 'number_of_filters': 64},

    {'type': 'CONV', 'filter_size': 3, 'padding': 0, 'stride': 1, 'number_of_filters': 64},
    {'type': 'CONV', 'filter_size': 3, 'padding': 0, 'stride': 1, 'number_of_filters': 64},
    {'type': 'CONV', 'filter_size': 3, 'padding': 0, 'stride': 1, 'number_of_filters': 64},
    # {'type': 'POOL', 'filter_size': 2, 'padding': 0, 'stride': 2, 'number_of_filters': 0},

    {'type': 'CONV', 'filter_size': 1, 'padding': 0, 'stride': 1, 'number_of_filters': 64}
]

print_info(SimpleCNN)
print_info(CNN)


