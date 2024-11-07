from multitasking.multifl import MultiFl, worker


def get_base_config():
    base_config = {'num_clients': 100, 'client_frac': 0.15, 'malicious_rate': 0.2,
                   'model_name': 'resnet18', 'data_name': 'cifar10', 'aggregate_type': 'flclaude',
                   'poison_type': 'trigger', 'poisoning_threshold': 5, 'num_epochs': 50,
                   'save_slogan': True, 'fl_print': True, 'sampling_stride': 2, 'alpha': 0.5,
                   'poison_probability': 0.3, 'pretrained': True}
    return base_config


def get_configs():
    base_config = get_base_config()
    configs = []
    # models: 'mobilenet', 'lenet_c3', 'resnet18', 'resnet34', 'resnet50', 'resnet101'
    models = ['lenet_c1']
    datas = ['mnist']
    # 'flame', 'small_flame', 'fltrust', 'small_fltrust, floss, fljoin'
    aggregates = ['fedavg', 'flame']
    # models: 'trigger', 'dba', 'semantic', 'sig', 'blended'
    poisons = ['trigger']
    # alpha
    alphas = [0.8]
    # malicious rate
    malicious_rates = [0.1]
    # poison probability
    poison_probability = [0.3]
    # pretrained
    pretrained_slog = [False]

    for model in models:
        for data in datas:
            for aggregate in aggregates:
                for poison in poisons:
                    for alpha in alphas:
                        for malicious_rate in malicious_rates:
                            for prob in poison_probability:
                                for pretrained in pretrained_slog:
                                    config = base_config.copy()
                                    config['model_name'] = model
                                    config['data_name'] = data
                                    config['aggregate_type'] = aggregate
                                    config['poison_type'] = poison
                                    config['alpha'] = alpha
                                    config['malicious_rate'] = malicious_rate
                                    config['poison_probability'] = prob
                                    config['pretrained'] = pretrained
                                    configs.append(config)

    return configs


def queue():
    configs = get_configs()
    for config in configs:
        try:
            worker(config.copy())
        except Exception as e:
            print(f"An error occurred: {e}")


def mul():
    configs = get_configs()
    num_configs = len(configs)
    batch = 10
    for i in range(0, num_configs, batch):
        multi_fl = MultiFl(configs[i:i + batch])
        multi_fl.multi_task()


def single():
    base_config = get_base_config()
    worker(base_config.copy())


if __name__ == '__main__':
    mode = 'mul'
    if mode == 'mul':
        mul()
    elif mode == 'single':
        single()
    elif mode == 'queue':
        queue()
    else:
        raise KeyError("mode must be either 'queue' or 'mul'")
