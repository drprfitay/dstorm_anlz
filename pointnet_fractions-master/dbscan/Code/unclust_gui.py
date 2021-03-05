__author__ = 'UriA12'

from tkinter import *
from tkinter import Tk
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from unclustered import main
pic_path = "SIA_image.gif"
fields_retsulc = ('Epsilon', 'Minimum Neighbors', 'Mini Epsilon', 'Mini Minimum Neighbors', 'Data Type', 'Red file','Green file', 'destination')
fields_filt = ('color','#red points', '#green points', 'density', 'size')

def go(ents):
    entries = ents[0]
    entries2 = ents[1]
    color = (entries2['color'].get())
    red_points = (entries2['#red points'].get())
    green_points = (entries2['#green points'].get())
    density = (entries2['density'].get())
    size = (entries2['size'].get())
    dest_path = (entries['destination'].get())
    red_name = (entries['Red file'].get())
    green_name = (entries['Green file'].get())
    data_type = (entries['Data Type'].get())
    epsilon = (int(entries['Epsilon'].get()))
    min_neighbors = (int(entries['Minimum Neighbors'].get()))
    mini_epsilon = (int(entries['Mini Epsilon'].get()))
    mini_min_neighbors = (int(entries['Mini Minimum Neighbors'].get()))
    main(data_type, epsilon, min_neighbors, mini_epsilon, mini_min_neighbors, green_name, red_name, dest_path, color, red_points, green_points, density, size)
    messagebox.showinfo("Work here is DONE!",\
    "If you wish choose something else.")

def makeform_left(root, fields):
   entries = {}
   cntr = 0
   for field in fields:
      if cntr != 4:
         row = Frame(root)
         lab = Label(row, width=22, text=field+": ", anchor='w')
         ent = Entry(row)
      if cntr == 0:
         ent.insert(0, "200")
      elif cntr == 1:
         ent.insert(0, "16")
      elif cntr == 2:
         ent.insert(0, "20")
      elif cntr == 3:
         ent.insert(0, "8")
      elif cntr == 4:
         row = Frame(root)
         lab = Label(row, width=22, text=field+": ", anchor='w')
         ent = ttk.Combobox(row)
         ent['values'] = ('2d', '3d', 'raw_2d', 'raw_3d', 'new_3d', "new_2d")
      row.pack(side=TOP, fill=X, padx=5, pady=5)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES, fill=X)
      entries[field] = ent
      cntr += 1
   return entries

def makeform_right(root, fields):
   entries = {}
   cntr = 0
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=22, text=field+": ", anchor='w', justify=LEFT)
      ent = Entry(row)
      if cntr == 0:
         ent = ttk.Combobox(row)
         ent['values'] = ('red', 'green', 'both')
         ent.insert(0, 'both')
      if cntr == 5:
         ent = ttk.Combobox(row)
         ent['values'] = ('yes', 'no', 'all')
         ent.insert(0, 'all')
      else:
         if cntr != 0 and cntr != 5 and cntr != 7 and cntr != 8 and cntr != 9:
            ent.insert(0, "MIN;MAX")
      row.pack(side=TOP, fill=Y, padx=5, pady=5)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, fill=X)
      entries[field] = ent
      cntr += 1
   return entries

def choose_dest(entries):
    path = filedialog.askdirectory()
    if path != "":
       entries['destination'].delete(0, END)
       entries['destination'].insert(0,path)

def choose_red(entries):
    path = filedialog.askopenfilename()
    if path != "":
       entries['Red file'].delete(0, END)
       entries['Red file'].insert(0,path)

def choose_green(entries):
    path = filedialog.askopenfilename()
    if path != "":
       entries['Green file'].delete(0, END)
       entries['Green file'].insert(0,path)

if __name__ == '__main__':
   root = Tk()
   root.wm_title("unclust_analysis!!:P")

   lab = Label(root, width=40, text="choose red file and green file and destination and more things :)", anchor='w', justify=LEFT)
   lab.pack(side=TOP)

   right_panel = Frame(root)
   left_panel = Frame(root)
   row_of_buttons = Frame(right_panel)

#left panel
   lab = Label(left_panel, width=40, text="Set the desired filters", anchor='w', justify=LEFT)
   lab.pack(side=TOP)
   panel = Label(left_panel)
   panel.pack(side = "bottom", fill = "both", expand = "yes")
   ents = makeform_left(left_panel, fields_retsulc)
   b_dest = Button(left_panel, text='Choose destination', command=(lambda e=ents: choose_dest(e)))
   b_dest.pack()
   b_g = Button(left_panel, text='Green File', command=(lambda e=ents: choose_green(e)))
   b_g.pack()
   b_r = Button(left_panel, text='Red File', command=(lambda e=ents: choose_red(e)))
   b_r.pack()
#right panel
   ents2 = makeform_right(right_panel, fields_filt)


  # low_panel = Frame(lef)
# buttons
   b_go = Button(row_of_buttons, text='GO!', command=(lambda e=(ents, ents2): go(e)))
   b_go.pack(side=RIGHT, padx=5, pady=5)
   left_panel.pack(side = LEFT)
   right_panel.pack(side = LEFT, expand = "yes")
   row_of_buttons.pack(side=BOTTOM)
   root.iconbitmap(r'SIA_icon.ico') # the icon
   root.mainloop()
