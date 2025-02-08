# Creating Name Lists
from operator import index

name = ["Raditya", "Saurav", "Bishal", "Binaya"]
print(name)

# Append Will repeat the name at last of the list
name.append("Raditya")
print(name)

# Remove will remove the name from the list
name.remove("Bishal")
print(name)

# Insert with the index will add a name in a particular index space in the list
name.insert(4, "Bishal")
print(name)

# More names added
moreNames = ["Sanjay", "Anjana"]
print(moreNames)

# Extend will add the 2nd list in the first list
name.extend(moreNames)
print(name)

# index shows the position of the items

name.index("Bishal")
print(name)







# # _________ Dictonary____________


season = {
    "Jan":"Winter",
    "Feb":"Winter",
    "Mar":"Spring",
    "Apr":"Spring",
    "May":"Winter",
    "June":"Summer",
    "July":"Summer",
    "Aug":"Summer",
    "Sep":"Fall",
    "Oct":"Fall",
    "Nov":"Fall",
    "Dec":"Fall",
}

month = input("Enter the name of the month:")

if month in season:
    print(f"The season {month} is {season[month]}.")

# ____________if-else_________________


number = int(input("Enter the number:"))

if number > 0:
    print("Number is positive.")

elif number < 0:
    print("Number is negative.")

else:
    print("Number is zero.")


userAge = int(input("Enter the age:"))
userCitizenship = input("Do you have a citizen? yes/no:")

if userAge >= 18 and userCitizenship == "yes":
    print("You are eligble. !YOU CAN VOTE!")
elif userAge < 18 or userCitizenship == "no":
    print("You are not eligble. !YOU CANNOT VOTE!")
else:
    print("Invalid")



#____________ For Loop___________

for num in range(1,11):
    product = num *2
    print(f"The product is:","2 *", num, "=", product)



#__________Function________

Calculate the radius of circle using function
import math

def areaCircle(radius):
    area = math.pi * (radius ** 2)
    return area

radius = int(input("Enter the radius of the circle: "))
area = areaCircle(radius)
print(f"Area of circle {radius} is {area}")


# Calculate the area of rectangle using function


def areaRectangle():
    length = float(input("Enter the length of the rectangle: "))
    width = float(input("Enter the width of the rectangle: "))
    area = length * width
    return area

# Call the function and display the result
rectangle_area = areaRectangle()
print(f"The area of the rectangle is: {rectangle_area}")


#___________Exception Handling____________

try:
    num = int(input("Enter a number: "))
    print(10 / num)
except ValueError:
    print("Please enter a number again.")
except ZeroDivisionError:
    print("You can't divide by zero.")



#CSV file and import pandas

import pandas as pd

df = pd.read_csv("Admission_Predict.csv")
print("First 10 data:")

print(df.head())

print("\n Last 7 data")

print(df.tail())
