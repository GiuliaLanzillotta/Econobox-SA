from data import test_location


if __name__ == "__main__":
    with open(test_location) as f:
        print("Working on ", test_location)
        # look at each line (=tweet)
        for l, line in enumerate(f):
            print(line[3:])
            if l > 10:
                exit(0)