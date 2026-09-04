import pandas as pd

csv_file = "attendance/Attendance.csv"
excel_file = "attendance/Attendance.xlsx"

df = pd.read_csv(csv_file)

df.to_excel(
    excel_file,
    index=False
)

print("Excel report created successfully!")
print(f"Saved as: {excel_file}")