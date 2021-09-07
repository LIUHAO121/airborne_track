import os, sys
sys.path.append(os.path.dirname(os.path.realpath(os.getcwd())))
sys.path.append(os.path.realpath(os.getcwd()))

from core.dataset import Dataset

if __name__ == "__main__":
    # option = input("Download full dataset (y/Y) or partial dataset (n/N)? ")
    # if option.lower() == 'y':
    #     print("Downloading full dataset...")
    #     partial=False
    # else:
    #     print("Downloading partial dataset...")
    #     partial=True

    dataset = Dataset(local_path='data/part3', s3_path='s3://airborne-obj-detection-challenge-training/part3/', prefix='part3', partial=True)
    # dataset.add(local_path='data/part2', s3_path='s3://airborne-obj-detection-challenge-training/part2/', prefix='part2')
    # dataset.add(local_path='data/part3', s3_path='s3://airborne-obj-detection-challenge-training/part3/', prefix='part3')

    # i = 1
    # all_flights = dataset.get_flight_ids()
    # for flight_id in all_flights:
    #     if "part1"+flight_id in os.listdir('/data/lhao/airmot/part1'):
    #         continue
    #     print("Downloading Flight#%s (%s/%s)..." % (flight_id, i, len(all_flights)))
    #     dataset.get_flight_by_id(flight_id).download()
    #     i += 1

    print("Download complete.")
