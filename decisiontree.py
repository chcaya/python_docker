#!/usr/bin/env python3

import pandas as pd
import numpy as np

def main():
    input_csv = pd.read_csv("input.csv")
    print(input_csv)


if __name__=="__main__":
    main()