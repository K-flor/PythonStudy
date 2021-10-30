
# tkinter 익히기
# 페이지 : https://realpython.com/python-gui-tkinter/

import tkinter as tk

widow = tk.Tk()

# widget 
# ------------------------- Label
greeting = tk.Label(text="Hello, Tkinter")
greeting.pack()
greeting2 = tk.Label(text="Python Rock~")
greeting2.pack()
la1 = tk.Label(text="FONT SETTING", foreground = 'white', background="black")
la1.pack()
la2 = tk.Label(text="FONT SETTING2", fg = "white", bg = "#34A2FE", width=15, height=10)
la2.pack()

# ------------------------- Button
button1 = tk.Button(text="Click Me!", width = 25, height = 5, bg = "blue", fg="yellow")
button1.pack()

widow.mainloop()

# ----------------------------- Entry (cmd에서 실행해야 원활히 get함수와 insert 등이 어떻게 사용되는지 알 수 있다)
window = tk.Tk()
label = tk.Label(text="name : ")
entry = tk.Entry()

label.pack()
entry.pack()

name = entry.get()
print(name)

entry.delete(0) # entry칸에 적힌 string의 0번째를 지움.
entry.insert(0, "python") # 0번 위치에서 부터 string을 낑겨넣음
entry.insert(tk.END, "12345") # tk.END는 string의 마지막부분을 아는 변수같음.

window.destroy() # close 버튼외에도 이 명령어를 사용해서 끌수있음.


# ------------------------------ Text
window = tk.Tk()
text_bar = tk.Text()
text_bar.pack()

window.mainloop()

text_bar.get("1.0") # text 위젯은 line number와 position에서 위치를 반환해줘야함.
text_bar.get("1.0", "1.4")
text_bar.get("1.0", tk.END)

text_bar.delete("1.0") # parameter 사용방법은 get과 동일.
text_bar.delete("1.0", "1.4")

text_bar.insert("1.5", "inserting state") # 첫 parameter는 넣은려는 위치, 두번째는 넣으려는 string

#-------------------------- Frame

window = tk.Tk()
frame = tk.Frame()

label1 = tk.Label(master = frame, text="maset is frame")
label1.pack()

label2 = tk.Label(text="master is window")
label2.pack()

frame.pack()
window.mainloop()

#-------
window = tk.Tk()

frame_a = tk.Frame()
label_a = tk.Label(master=frame_a, text="Im in Frame A")
label_a.pack()

frame_b = tk.Frame()
label_b = tk.Label(master=frame_b, text="Im in Frame B")
label_b.pack()

frame_b.pack()
frame_a.pack()

window.mainloop()


# ------ frame setting

border_effects = {"flat":tk.FLAT, 
                  "sunken": tk.SUNKEN, 
                  "raised" : tk.RAISED, 
                  "groove":tk.GROOVE, 
                  "ridge":tk.RIDGE}

window = tk.Tk()

for relief_name, relief in border_effects.items():
    frame = tk.Frame(master=window, relief = relief, borderwidth=5)
    frame.pack(side=tk.LEFT)
    label = tk.Label(master=frame, text=relief_name)
    label.pack()

window.mainloop()


# ----------------------------------- Controlling Layout with Geometry Managers
#-------------------------- pack()
# frame 크기가 각각
window = tk.Tk()

frame1 = tk.Frame(master=window, width=100, height=100, bg="red")
frame1.pack()

frame2 = tk.Frame(master=window, width=50, height=50, bg="yellow")
frame2.pack()

frame3 = tk.Frame(master=window, width=25, height=25, bg="blue")
frame3.pack()

window.mainloop()

# frame 크기가 window에 따라 다름. 단, 가로방향으로만
window = tk.Tk()

frame1 = tk.Frame(master=window, height=100, bg="red")
frame1.pack(fill=tk.X)

frame2 = tk.Frame(master=window, height=50, bg="yellow")
frame2.pack(fill=tk.X)

frame3 = tk.Frame(master=window, height=25, bg="blue")
frame3.pack(fill=tk.X)

window.mainloop()

# frame 크기가 window에 따라 다름. 단, 세로방향으로
window = tk.Tk()

frame1 = tk.Frame(master=window, width=200, height=100, bg="red")
frame1.pack(fill=tk.Y, side=tk.LEFT)

frame2 = tk.Frame(master=window, width=100, bg="yellow")
frame2.pack(fill=tk.Y, side=tk.LEFT)

frame3 = tk.Frame(master=window, width=50, bg="blue")
frame3.pack(fill=tk.Y, side=tk.LEFT)

window.mainloop()

# frame 크기가 window에 따라 다름. 가로, 세로 모두!
window = tk.Tk()

frame1 = tk.Frame(master=window, width=200, height=100, bg="red")
frame1.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

frame2 = tk.Frame(master=window, width=100, bg="yellow")
frame2.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

frame3 = tk.Frame(master=window, width=50, bg="blue")
frame3.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

window.mainloop()

#-------------------------- place ---> 이건 위치를 지정해야해서, 알아야 잘할듯

window = tk.Tk()

frame = tk.Frame(master=window, width=150, height=150)
frame.pack()

label1 = tk.Label(master=frame, text="I'm at (0, 0)", bg="red")
label1.place(x=0, y=0)

label2 = tk.Label(master=frame, text="I'm at (75, 75)", bg="yellow")
label2.place(x=75, y=75)

window.mainloop()

# ---------------------- grid
# grid 예제 1
window = tk.Tk()

for i in range(3):
    for j in range(3):
        frame = tk.Frame(
            master=window,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame.grid(row=i, column=j)
        label = tk.Label(master=frame, text=f"Row {i}\nColumn {j}")
        label.pack()

window.mainloop()

# grid 예제 2, padx와 pady를 추가한 grid 사용
window = tk.Tk()

for i in range(3):
    for j in range(3):
        frame = tk.Frame(
            master=window,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame.grid(row=i, column=j, padx=5, pady=5)

        label = tk.Label(master=frame, text=f"Row {i}\nColumn {j}")
        label.pack(padx=5, pady=5)

window.mainloop()


# grid 예제 3, 예제 2는 창의 크기가 바뀌어도 grid의 크기가 유지된다. 예제 3은 expand에 반응하는 윈도우
window = tk.Tk()

for i in range(3):
    window.columnconfigure(i, weight=1, minsize=75)
    window.rowconfigure(i, weight=1, minsize=50)

    for j in range(0, 3):
        frame = tk.Frame(
            master=window,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame.grid(row=i, column=j, padx=5, pady=5)

        label = tk.Label(master=frame, text=f"Row {i}\nColumn {j}")
        label.pack(padx=5, pady=5)

window.mainloop()


# grid 예제 4
# sticky 
#   'n', 'N' : align to the top-center of the cell
#   'e', 'E' : align to the right-center of the cell
#   's', 'S' : align to the bottom-center of the cell
#   'w', 'W' : align to the left-center of the cell
window = tk.Tk()

window.rowconfigure(0, minsize=50)
window.columnconfigure([0, 1, 2, 3], minsize=50)

label1 = tk.Label(text="1", bg="black", fg="white")
label2 = tk.Label(text="2", bg="black", fg="white")
label3 = tk.Label(text="3", bg="black", fg="white")
label4 = tk.Label(text="4", bg="black", fg="white")

label1.grid(row=0, column=0)
label2.grid(row=0, column=1, sticky="ew")
label3.grid(row=0, column=2, sticky="ns")
label4.grid(row=0, column=3, sticky="nsew")

window.mainloop()


# Exercise : Create an address entry form
window = tk.Tk()

texts = ["First Name", "Last Name", "Address Line1", "Address Line2", "city", "State", "Postal Code", "Country"]

for i in range(8):
    for j in range(2):
        frame = tk.Frame(master=window )
        frame.grid(row=i, column=j)
        
        if j % 2 == 0 :
            label = tk.Label(master=frame, text=f'{texts[i]} :')
            label.pack()
        else :
            entry = tk.Entry(master=frame)
            entry.pack()
    
window.mainloop()


# ----------------------------------- Building a Text Editor
from tkinter.filedialog import askopenfile

window = tk.Tk()
window.title("Simple Text Editor")

window.rowconfigure(0, minsize=800, weight=1) 
window.columnconfigure(1, minsize=800, weight=1)

txt_editor = tk.Text(window)
ft_button = tk.Frame(window)
btn_open = tk.Button(ft_button, text="Open")
btn_save = tk.Button(ft_button, text="Save As...")

btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_save.grid(row=1, column=0, sticky="ew", padx=5)
ft_button.grid(row=0, column=0, sticky="ns")
txt_editor.grid(row=0, column=1, sticky="nsew")

window.mainloop()

def open_file():
    """Open a file for editing."""
    filepath = askopenfilename(
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if not filepath:
        return
    txt_edit.delete("1.0", tk.END)
    with open(filepath, "r") as input_file:
        text = input_file.read()
        txt_edit.insert(tk.END, text)
    window.title(f"Simple Text Editor - {filepath}")