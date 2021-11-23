from datetime import datetime, timedelta

import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
import numpy as np


NUM_DAYS = [
    59,
    53,
    57,
    60,
    36,
    49,
    28,
    56,
    60,
    26,
    54,
    94,
    488,
    30,
    296,
    59,
    221,
    19,
    30,
    63,
    9,
    26,
    30,
    13,
    57,
    36,
    26,
    60,
    28,
    29,
]

csv_file = pd.read_csv("/home/sharare/PycharmProjects/FederatedLearning_Caching/datasets/avaiable data4.csv")

df = pd.DataFrame(csv_file)

# fig = px.timeline(df,
#                   x_start="Start day",
#                   x_end="Target day",
#                   y="Home",
#                   range_x=["2020-01-01", "2020-01-31"],
#                   range_y=[1, 30],
#                   )

df.columns = ['Task', 'Start', 'Finish', 'Resource']

# print(df['Start'].dtype)
df['Start'] = df['Start'].astype('datetime64[ns]')
df['Finish'] = df['Finish'].astype('datetime64[ns]')

for i in range(43):
    print(df.loc[i]['Finish'])

    home_id = int(df.loc[i]['Task']) - 1
    df.at[i, 'Finish'] = min(df.loc[i]['Start'] + timedelta(NUM_DAYS[home_id]), datetime(year=2020, month=4, day=30))
    if df.at[i, 'Resource']:
        # df.at[i, 'Finish'] = (df.loc[i]['Start'] + timedelta(NUM_DAYS[home_id]))
        df.at[i, 'Finish'] = min(df.loc[i]['Start'] + timedelta(NUM_DAYS[home_id]), datetime(year=2020, month=1, day=16))
    # if df.at[i, 'Resource']:

    #     df.at[i, 'Finish'] = min(df.loc[i]['Start'] + timedelta(NUM_DAYS[i]), datetime(year=2020, month=1, day=16))

    # df.at[i, 'Finish'] = datetime(year=2020, month=3, day=3)
    # df.set_value('Finish', i, df.loc[i]['Start'] + timedelta(NUM_DAYS[i]))
    # df[i]['Finish'] = df.loc[i]['Start'] + timedelta(NUM_DAYS[i])
    # print(df.loc[i]['Finish'])

print(df.head())
# z.head()
# df['Finish'] = df['Start'] + z
# df.head()

df['Resource'] = ['started' if i else 'Not_started' for i in df['Resource']]
critical_colors = {'started': 'rgb(202, 47, 85)', 'Not_started': 'rgb(107, 127, 135)'}

# fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
# fig = ff.create_gantt(df, colors=critical_colors, index_col='Resource', title='Gantt Chart',
#                               bar_width=0.4, showgrid_x=True, showgrid_y=True)
fig = ff.create_gantt(df,
                      colors=critical_colors,
                      index_col='Resource',
                      title='Gantt Chart',
                      bar_width=0.4,
                      showgrid_x=True,
                      showgrid_y=True,
                      height=1000,
                      group_tasks=True,
                      )

fig.update_layout(
    title="Poisson arrival",
    xaxis_title="Day",
    yaxis_title="Home",
    font=dict(
        family="arial",
        size=14,
        # color="RebeccaPurple"
    )
)

fig.show()
