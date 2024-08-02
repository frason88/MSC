import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('msc_final_data_v1.csv')

def get_top_5_percentages(data):
    course_counts = data['course_name'].value_counts()
    total_count = course_counts.sum()
    top_5_percentages = (course_counts.head(5) / total_count * 100).round(2)
    return top_5_percentages

pre_covid_data = df[df['pandemic'] == 'Pre-COVID']
post_covid_data = df[df['pandemic'] == 'Post-COVID']

pre_covid_percentages = get_top_5_percentages(pre_covid_data)
post_covid_percentages = get_top_5_percentages(post_covid_data)

print("Pre-COVID Top 5 Course Percentages:")
print(pre_covid_percentages)
print("\nPost-COVID Top 5 Course Percentages:")
print(post_covid_percentages)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

def create_pie_chart(ax, data, title):
    ax.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(title)

create_pie_chart(ax1, pre_covid_percentages, 'Pre-COVID Top 5 Courses')
create_pie_chart(ax2, post_covid_percentages, 'Post-COVID Top 5 Courses')

plt.tight_layout()
plt.show()