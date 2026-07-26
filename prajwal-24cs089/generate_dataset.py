import csv
import random

num_min_range = 0
num_max_range = 1000
fakecount =0
correct_count = 0

total_count = 10000

with open("train_data.csv" , "w", newline="") as f:
    writer = csv.writer(f)
 

    for i in range( total_count):
        x = random.randint( num_min_range ,num_max_range )
        y = random.randint( num_min_range ,num_max_range )
        # row_list = []

        chance = random.randint(0,100)

        if chance <= 95:

            writer.writerow( [x,y , x+y] )
            correct_count +=1
        else:

            fakecount += 1
            writer.writerow( [x, y , random.randint(1, num_max_range)] )

print(f"Total entries = {total_count} Wrong entries = {fakecount} , correct entries = {correct_count } ")
      
print(f"% of wrong entries = {(fakecount/total_count)*100 }")
