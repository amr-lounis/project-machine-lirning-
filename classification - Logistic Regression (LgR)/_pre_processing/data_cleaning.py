import pandas as pd

_path_data = '../data/'
_path = _path_data+'wifi_localization.txt'

columns = [
    'wifi_01',
    'wifi_02',
    'wifi_03',
    'wifi_04',
    'wifi_05',
    'wifi_06',
    'wifi_07',
    'room']

#read file
data = pd.read_csv(_path, header=None, names=columns , sep='\t')

# rescaling data but last one column
for i in range(0,len(columns)-1):
    data[columns[i]] = (data[columns[i]] - data[columns[i]].min()) / (data[columns[i]].max()-data[columns[i]].min())


data.to_csv(_path_data+'data_clean.csv', index=False, encoding='utf-8-sig')
    
#this function create dataset for one vs rest model
def select_one_vs_rest(_data,_room):
    tmp =  _data.copy()
    tmp.loc[tmp['room'] != _room , 'room'] = 0
    tmp.loc[tmp['room'] == _room , 'room'] = 1
    tmp.to_csv(_path_data + f'data_clean_one_vs_rest_{_room}.csv',
               index=False, encoding='utf-8-sig') 
	
# create dataset for one vs rest model
select_one_vs_rest(data,1)
select_one_vs_rest(data,2)
select_one_vs_rest(data,3)
select_one_vs_rest(data,4)

#this function create dataset for one vs one model
# def select_one_vs_one(_data,_room1,_room2):
#     _rooms_selected = [_room1,_room2]
#     for i in range(1,5):
#         if(i not in _rooms_selected):
#             room_delete= _data[_data['room']  == i ].index
#             _data = _data.drop(room_delete)
#     _data['room'] = (_data['room']==_rooms_selected[0]).astype(int)
#     _data.to_csv(_path_data + f'data_clean_one_vs_one_{_room1}vs{_room2}.csv', index=False, encoding='utf-8-sig')
	
# create dataset for one vs one model
# select_one_vs_one(data,1,2)
# select_one_vs_one(data,1,3)
# select_one_vs_one(data,1,4)
# select_one_vs_one(data,2,3)
# select_one_vs_one(data,2,4)
# select_one_vs_one(data,3,4)
