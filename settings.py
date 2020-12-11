MAPPED = True
USE_TIME = True
USE_DAY = True
NUMBER_OF_DAYS = 7
NUMBER_OF_HOURS = 24

if MAPPED:
    NUMBER_OF_ACTIVITIES = 10
else:
    NUMBER_OF_ACTIVITIES = 47

TOTAL_NUM_OF_FEATURES = NUMBER_OF_ACTIVITIES  # 10 for mapped
if USE_DAY:
    TOTAL_NUM_OF_FEATURES += NUMBER_OF_DAYS
if USE_TIME:
    TOTAL_NUM_OF_FEATURES += NUMBER_OF_HOURS


NUM_ROUNDS = 20
BATCH_SIZE = 64
HISTORY_SIZE = 24
BUFFER_SIZE = 1000
SEQ_LENGTH = 50
STEPS = 2

CONFUSION_MATRIX_DIR = './confusion_matrices/'

LIST_OF_ACTIVITIES = ['Step_Out', 'Other_Activity', 'Toilet', 'Personal_Hygiene', 'Watch_TV', 'Leave_Home',
                      'Enter_Home', 'Take_Medicine', 'Work', 'Exercise', 'Work_On_Computer',
                      'Dress', 'Evening_Meds', 'Sleep_Out_Of_Bed', 'Wake_Up', 'Read', 'Morning_Meds',
                      'Cook_Breakfast', 'Nap', 'Laundry', 'r1.Sleep', 'r1.Cook_Breakfast',
                      'r2.Personal_Hygiene', 'r2.Eat_Breakfast', 'r2.Dress',
                      'Eat_Breakfast', 'Bathe', 'Phone', 'Cook_Lunch', 'Eat_Lunch', 'Wash_Lunch_Dishes',
                      'Relax', 'Work_At_Desk',
                      'Drink',
                      'Go_To_Sleep', 'Sleep', 'Bed_Toilet_Transition', 'Groom', 'Cook',
                      'Wash_Breakfast_Dishes',
                      'Wash_Dishes', 'Cook_Dinner', 'Eat', 'Eat_Dinner', 'Entertain_Guests',
                      'Wash_Dinner_Dishes',
                      'Work_At_Table']

LIST_OF_MAPPED_ACTIVITIES = ['PERSONAL_HEALTH_AND_HYGIENE', 'PERSONAL_HYGIENE', 'RELAX', 'LEAVE_HOME', 'ENTER_HOME',
                             'HEALTH', 'WORK', 'GROOMING', 'REST', 'COOK',
                             'CHORES', 'EAT', 'SOCIAL', 'WASH_DISHES', 'DRINK'
                             ]

LIST_OF_MAPPED_2_ACTIVITIES = ['PERSONAL_HEALTH_AND_HYGIENE', 'RELAX', 'LEAVE_HOME', 'ENTER_HOME',
                               'WORK', 'REST',
                               'CHORES', 'EAT', 'SOCIAL', 'DRINK'
                               ]

MAP_OF_ACTIVITIES = {
    'Step_Out': 'NOT_TRACKED',
    'Other_Activity': 'NOT_TRACKED',
    'Toilet': 'PERSONAL_HEALTH_AND_HYGIENE',
    'Personal_Hygiene': 'PERSONAL_HYGIENE',
    'Watch_TV': 'RELAX',
    'Leave_Home': 'LEAVE_HOME',
    'Enter_Home': 'ENTER_HOME',
    'Take_Medicine': 'HEALTH',
    'Work': 'WORK',
    'Exercise': 'PERSONAL_HEALTH_AND_HYGIENE',
    'Work_On_Computer': 'WORK',
    'Dress': 'GROOMING',
    'Evening_Meds': 'HEALTH',
    'Sleep_Out_Of_Bed': 'REST',
    'Wake_Up': 'REST',
    'Read': 'RELAX',
    'Morning_Meds': 'HEALTH',
    'Cook_Breakfast': 'COOK',
    'Nap': 'REST',
    'Laundry': 'CHORES',
    'r1.Sleep': 'REST',
    'r1.Cook_Breakfast': 'COOK',
    'r2.Personal_Hygiene': 'PERSONAL_HYGIENE',
    'r2.Eat_Breakfast': 'EAT',
    'r2.Dress': 'GROOMING',
    'Eat_Breakfast': 'EAT',
    'Bathe': 'PERSONAL_HYGIENE',
    'Phone': 'SOCIAL',
    'Cook_Lunch': 'COOK',
    'Eat_Lunch': 'EAT',
    'Wash_Lunch_Dishes': 'WASH_DISHES',
    'Relax': 'RELAX',
    'Work_At_Desk': 'WORK',
    'Drink': 'DRINK',
    'Go_To_Sleep': 'REST',
    'Sleep': 'REST',
    'Bed_Toilet_Transition': 'NOT_TRACKED',
    'Groom': 'GROOMING',
    'Cook': 'COOK',
    'Wash_Breakfast_Dishes': 'WASH_DISHES',
    'Wash_Dishes': 'WASH_DISHES',
    'Cook_Dinner': 'COOK',
    'Eat': 'EAT',
    'Eat_Dinner': 'EAT',
    'Entertain_Guests': 'SOCIAL',
    'Wash_Dinner_Dishes': 'WASH_DISHES',
    'Work_At_Table': 'WORK',
    'COOK': 'CHORES',
    'WASH_DISHES': 'CHORES',
    # 'EAT': 'FOOD',
    'HEALTH': 'PERSONAL_HEALTH_AND_HYGIENE',
    'GROOMING': 'PERSONAL_HEALTH_AND_HYGIENE',
    'PERSONAL_HYGIENE': 'PERSONAL_HEALTH_AND_HYGIENE'
}

ACT2IDX = {u: i for i, u in enumerate(LIST_OF_ACTIVITIES)}

MAPPED_ACT2IDX = {u: i for i, u in enumerate(LIST_OF_MAPPED_ACTIVITIES)}

MAPPED_2_ACT2IDX = {u: i for i, u in enumerate(LIST_OF_MAPPED_2_ACTIVITIES)}

SAMPLING = False

AVAILABLE_CLIENTS = {2: [103],
                     5: [103, 129],
                     6: [103, 129, 111, 114],
                     7: [103, 129, 111, 114, 127],
                     9: [103, 129, 111, 114, 127, 125],
                     12: [103, 129, 111, 114, 127, 125, 112, 128],
                     13: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123],
                     14: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106],
                     15: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117],
                     16: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109],
                     19: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115],
                     22: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124],
                     23: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121],
                     24: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102],
                     26: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130],
                     27: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130,
                          107],
                     34: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130,
                          107, 105],
                     37: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130,
                          107, 105, 119],
                     42: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130,
                          107, 105, 119, 120],
                     45: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130,
                          107, 105, 119, 120, 110],
                     46: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130,
                          107, 105, 119, 120, 110, 108],
                     47: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130,
                          107, 105, 119, 120, 110, 108, 126, 104],
                     48: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130,
                          107, 105, 119, 120, 110, 108, 126, 104, 101],
                     56: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130,
                          107, 105, 119, 120, 110, 108, 126, 104, 101, 122],
                     61: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130,
                          107, 105, 119, 120, 110, 108, 126, 104, 101, 122, 116],
                     62: [103, 129, 111, 114, 127, 125, 112, 128, 118, 123, 106, 117, 109, 115, 124, 121, 102, 130,
                          107, 105, 119, 120, 110, 108, 126, 104, 101, 122, 116, 113]}

LIST_OF_DAYS_OF_UPDATES = [2, 5, 6, 7, 9, 12, 13, 14, 15, 16, 19, 22, 23, 24, 26, 27, 34, 37, 42, 45, 46, 47, 48, 56,
                           61, 62]

CLIENT_START_DAY = {
    103: 2,
    129: 5,
    111: 6,
    114: 6,
    127: 7,
    125: 9,
    112: 12,
    128: 12,
    118: 13,
    123: 13,
    106: 14,
    117: 15,
    109: 16,
    115: 19,
    124: 22,
    121: 23,
    102: 24,
    130: 26,
    107: 27,
    105: 34,
    119: 37,
    120: 42,
    110: 45,
    108: 46,
    126: 47,
    104: 47,
    101: 48,
    122: 56,
    116: 61,
    113: 62
}
# 3.0	2020-01-02
# 29.0	2020-01-05
# 14.0	2020-01-06
# 11.0	2020-01-06
# 27.0	2020-01-07
# 25.0	2020-01-09
# 28.0	2020-01-12
# 12.0	2020-01-12
# 18.0	2020-01-13
# 23.0	2020-01-13
# 6.0	2020-01-14
# 17.0	2020-01-15
# 9.0	2020-01-16
# 15.0	2020-01-19
# 24.0	2020-01-22
# 21.0	2020-01-23
# 2.0	2020-01-24
# 30.0	2020-01-26
# 7.0	2020-01-27
# 5.0	2020-02-03
# 19.0	2020-02-06
# 20.0	2020-02-11
# 10.0	2020-02-14
# 8.0	2020-02-15
# 26.0	2020-02-16
# 4.0	2020-02-17
# 1.0	2020-02-18
# 22.0	2020-02-26
# 16.0	2020-03-02
# 13.0	2020-03-03

# export PYTHONPATH=$PYTHONPATH:/home/sharare/PycharmProjects/FederatedLearning_Caching
