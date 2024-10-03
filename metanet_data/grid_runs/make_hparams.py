import csv
import random

def main():
    with open('hparams.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['lr', 'weight_decay', 'label_smoothing', 'epochs'])

        for idx in range(10000):
            lr = .5 * 10**(-random.uniform(0, 2))
            weight_decay = 10**(-random.uniform(1, 5))
            label_smoothing = random.uniform(0, .2)
            epochs = random.randint(10, 40)
            writer.writerow([lr, weight_decay, label_smoothing, epochs])

if __name__ == '__main__':
    main()
