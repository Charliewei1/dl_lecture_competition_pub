import numpy as np
import sys


def load_arrays(file_names):
    return [np.load(file_name) for file_name in file_names]


def majority_vote(arrays):
    result = []
    for i in range(len(arrays[0])):
        counts = {}
        for arr in arrays:
            value = arr[i]
            counts[value] = counts.get(value, 0) + 1

        max_count = max(counts.values())
        majority_values = [key for key, count in counts.items() if count == max_count]

        if len(majority_values) == 1:
            result.append(majority_values[0])
        else:
            for arr in arrays:
                if arr[i] in majority_values:
                    result.append(arr[i])
                    break
            else:
                result.append(
                    arrays[0][i]
                )  # Default to first array (a) if all different

    return np.array(result)


def main():
    if len(sys.argv) != 6:
        print(
            "Usage: python ensemble.py file1.npy file2.npy file3.npy file4.npy file5.npy"
        )
        sys.exit(1)

    file_names = sys.argv[1:]
    arrays = load_arrays(file_names)

    result = majority_vote(arrays)

    np.save("submit_ensemble_result.npy", result)


if __name__ == "__main__":
    main()
