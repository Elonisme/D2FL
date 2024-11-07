from result.plot import get_filenames_in_directory, plot, load_csv

if __name__ == '__main__':
    directory = "save/csv/"
    model_targets = ['resnet18']
    attack_targets = ['trigger']
    for model_target in model_targets:
        for attack_target in attack_targets:
            keywords = [model_target, "cifar10", attack_target, '50', "0.1", "0.3", "0.15", "True"]
            #  ["deepsight", "flclaude", "assembly", "rflbat", "fljoin", "old", "floss", "fltrust", "krum", "small_flame", "flame"]
            not_keywords = ["0.2", "deepsight", "flclaude", "assembly", "rflbat", "fljoin", "median", "old", "floss", "krum", "small_flame"]

            filenames = get_filenames_in_directory(directory, keywords, not_keywords)
            # filenames = get_filenames_in_directory(directory, keywords)

            result_ma = {}
            result_ba = {}
            result_loss = {}
            result_time = {}
            epoch = 0

            split_filename = filenames[0].split("-")
            model_name = model_target
            data_name = split_filename[1]
            img_path = f"save/img/all/{model_name}-{data_name}-{attack_target}"

            for filename in filenames:
                split_filename = filename.split("-")
                model_name = split_filename[0]
                data_name = split_filename[1]
                attack_name = split_filename[2]
                defense_name = split_filename[3]
                img_path += f"-{defense_name}"
                data = load_csv(directory, filename)
                num_data = len(data)
                if num_data == 3:
                    ma, ba, loss = data
                elif num_data == 4:
                    ma, ba, loss, time = data
                    result_time[str(defense_name)] = time
                elif num_data == 6:
                    ma, ba, loss, time, _, _ = data
                    result_time[str(defense_name)] = time
                else:
                    raise ValueError
                epoch = [i * 5 for i in range(0, len(loss))]
                result_ma[str(defense_name)] = ma
                result_ba[str(defense_name)] = ba
                result_loss[str(defense_name)] = loss

            plot(epoch, result_ma, img_path + '-ma', title='Main Task Accuracy', y_label="ma")
            plot(epoch, result_ba, img_path + '-ba', title='Backdoor Task Accuracy', y_label="ba")
            plot(epoch, result_loss, img_path + '-loss', title='Loss Value', y_label="loss")
            if len(result_time) > 0:
                plot(epoch, result_time, img_path + '-time', title='Time', y_label="time")
