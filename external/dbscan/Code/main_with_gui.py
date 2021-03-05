from tkinter import *
from tkinter import Tk
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from parse_super_detect import go

pic_path = "SIA_image.gif"
fields = ('Epsilon', 'Minimum Neighbors', 'Mini Epsilon', 'Mini Minimum Neighbors', 'Data Type', 'Mode')
f_fields = ('color', '#points', '#red points', '#green points', 'density', 'colocalized', 'angle x', 'angle y', 'size')


def gogo(entries):
    #   name = (entries['Session Name'].get())
    epsilon = (int(entries['Epsilon'].get()))
    min_neighbors = (int(entries['Minimum Neighbors'].get()))
    mini_epsilon = (int(entries['Mini Epsilon'].get()))
    mini_min_neighbors = (int(entries['Mini Minimum Neighbors'].get()))
    prot_mode = 2 if entries['Mode'].get() == "2 protein" else 1
    d_type = (entries['Data Type'].get())
    path = filedialog.askdirectory()

    f_color = (entries['color'].get())
    f_points = (entries['#points'].get())
    f_red_points = (entries['#red points'].get())
    f_green_points = (entries['#green points'].get())
    f_density = (entries['density'].get())
    f_coloc = (entries['colocalized'].get())
    f_x_angle = (entries['angle x'].get())
    f_y_angle = (entries['angle y'].get())
    f_size = (entries['size'].get())

    print(path)
    messagebox.showinfo("Work in progress",
                        "Please wait till' it's done... You'll get a message (for now just click OK).")
    go(epsilon, min_neighbors, mini_epsilon, mini_min_neighbors, d_type, path, prot_mode, f_color, f_points, f_red_points,
       f_green_points, f_density, f_coloc, f_x_angle, f_y_angle, f_size)
    messagebox.showinfo("Work is DONE!", "You may now enter another session folder.")


def makeform(root, fields):
    entries = {}
    cntr = 0
    for field in fields:
        if cntr != 4:
            row = Frame(root)
            lab = Label(row, width=22, text=field + ": ", anchor='w')
            ent = Entry(row)
        if cntr == 0:
            ent.insert(0, "200")
        elif cntr == 1:
            ent.insert(0, "16")
        elif cntr == 2:
            ent.insert(0, "50")
        elif cntr == 3:
            ent.insert(0, "8")
        elif cntr == 4:
            row = Frame(root)
            lab = Label(row, width=22, text=field + ": ", anchor='w')
            ent = ttk.Combobox(row)
            ent['values'] = ('2d', '3d', 'raw_2d', 'raw_3d', 'new_3d', 'new_2d')
            ent.insert(0, "new_2d")
        elif cntr == 5:
            row = Frame(root)
            lab = Label(row, width=22, text=field + ": ", anchor='w')
            ent = ttk.Combobox(row)
            ent['values'] = ('1 protein', '2 protein')
            ent.insert(0, "2 protein")
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries[field] = ent
        cntr += 1
    return entries
def make_f_form(root, fields):
    entries = {}
    cntr = 0
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=22, text=field + ": ", anchor='w', justify=LEFT)
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
            if cntr != 0 and cntr != 5 and cntr != 9 and cntr != 10 and cntr != 11:
                ent.insert(0, "MIN;MAX")
        row.pack(side=TOP, fill=Y, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, fill=X)
        entries[field] = ent
        cntr += 1
    return entries


if __name__ == '__main__':
    root = Tk()
    root.wm_title("Retsulc (ver 0.2) - Get clusters!")
    img = Image.open(pic_path)
    photo = ImageTk.PhotoImage(img)

    right_panel = Frame(root)
    left_panel = Frame(root)
    # row_of_buttons = Frame(right_panel)

    panel = Label(left_panel, image=photo)
    panel.pack(side=BOTTOM, fill="both", expand="yes")
    # listbox = Listbox(right_panel, width=100, height=30)
    # listbox.pack(side=RIGHT, fill=BOTH)

    ents = makeform(left_panel, fields)
    f_ents = make_f_form(right_panel, f_fields)
    ents.update(f_ents)
    b2 = Button(left_panel, text='GO!', command=(lambda e=ents: gogo(e)))
    b2.pack(side=BOTTOM, padx=5, pady=5)

    # row_of_buttons.pack(side=BOTTOM, padx=5, pady=5)
    left_panel.pack(side=LEFT, expand="yes")
    right_panel.pack(side=RIGHT, expand="yes")

    root.iconbitmap(r'SIA_icon.ico')
    root.mainloop()
