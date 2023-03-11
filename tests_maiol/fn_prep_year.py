import numpy as np

#import matplotlib.pyplot as plt
import netCDF4 as nc
import glob
import numpy.ma as ma
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os

#def get_precipitation():
def get_time_point(time_str):
    str_time = time_str.split("/")
    str_time = str_time[-1]
    list_time = str_time.split("_")
    point = list_time[0]
    year = list_time[1]
    day = list_time[2]
    day = day[:len(day)-3]
    #print(point, year, day)
    return point, year, day

def generate_plot(dir, filter_point):
    '''GETTING FILES AND DATA'''
    # Get list of all files in a given directory sorted by name
    list_of_paths = sorted(filter( os.path.isfile, glob.glob(dir+'*.nc', recursive=True)))

    time_target = {}
    for point in list_of_paths:
        point_str, year, day = get_time_point(point)
        point_data = nc.Dataset(point)
        #print(point_str)
        target = float(point_data['target'][:].data)
        #print(target)
        if point_str not in time_target:
            time_target[point_str] = {}
        if year not in time_target[point_str]:
            time_target[point_str][year] = {}
            time_target[point_str][year]['days'] = []
            time_target[point_str][year]['prep'] = []
        time_target[point_str][year]['days'].append(day)
        time_target[point_str][year]['prep'].append(target)

    #print(len(time_target['p1']['2000']['days']))
    #print(len(time_target['p1']['2000']['prep']))

    ''' CHANGE FILTER POIIIIINT!'''
    filter_point = filter_point

    '''Highest value for prep/day in X year'''
    print('Highest prep '+str(max(time_target[filter_point]['2000']['prep'])))

    '''Count how many 0s'''
    count = time_target[filter_point]['2000']['prep'].count(0.0000000000)
    print('Days with 0 prep:', count)

    fig = plt.figure(figsize=(11.0, 8.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    #si filter point es p1, hará el loop y el plot para todos los años en este point
    #p.ej, el fichero p1_2000_001, -> p1 en año 2000 dia 1 etc...
    for year in time_target[filter_point]:
        days =  time_target[filter_point][year]['days'] #day number
        prep = time_target[filter_point][year]['prep'] #prep value
        ax.plot(days, prep, label=year)

    # label the axes and title the plot
    ax.set_xlabel('months')
    ax.set_ylabel('precipitation (mm)')
    ax.set_title('Precipitation/year in '+filter_point)

    #la conversion de dias a months:
    #https://stackoverflow.com/questions/62360855/convert-x-axis-from-days-to-month-in-matplotlib
    month_starts = [0, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306, 336]
    #0 o 1 para Jan..
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names)

    ax.legend()

    # saving the plot
    fig.savefig('./plots/year-precipitation-'+filter_point+'.png', bbox_inches='tight')

def main():
    dir = "./data_sample_victor/"
    #A CAMBIAR EN FUNCIÓN DEL NOMBRE DEL FICHERO!!
    filter_point = "p2"
    generate_plot(dir, filter_point)

if __name__ == "__main__":
    #main con parametros de conf
    main()
