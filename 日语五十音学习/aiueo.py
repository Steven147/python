from random import choice
import tkinter

#initialize
value = '@'
mystr = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
mystr2 = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
romaji = ['a','i','u','e','o','ka','ki','ku','ke','ko','sa','si','su','se','so','ta','chi','tsu','te','to','na','ni','nu','ne','no','ha','hi','hu','he','ho','ma','mi','mu','me','mo','ya','yu','yo','ra','ri','ru','re','ro','wa','wo','n']
show = 0
show2 = 0

def showVar(flag=True):
    global show
    if flag:
        var.set(romaji[value])
    else:
        var.set('')         

def showVar2(flag=True):
    if flag:
        if show2 == 0:
            var2.set(mystr[value]+' '+mystr2[value])
        elif show2 == 1:
            var2.set(mystr[value])
        else:
            var2.set(mystr2[value])
    else:
        var2.set('')      


#botton1
def botton1():
    global value, show
    value = choice(range(0,46,1))

    #show result
    if(show == 0):
        showVar()
        showVar2()
    elif(show == 1): 
        showVar()
        showVar2(False)
    else:
        showVar(False)
        showVar2()

#botton2
def botton2():
    showVar()
    showVar2()

#botton3
def botton3():
    global show
    show = (show + 1) % 3
    if(show == 0): 
        var3.set("Learning")
        var2.set(mystr[value]+' '+mystr2[value])
    elif(show == 1):
        var3.set('Character Testing')
        var2.set('')
    else:
        var3.set('Pronouce Testing')
        var.set('')

#botton4
def botton4():
    global show2
    show2 = (show2 + 1) % 3
    if(show２ == 0): 
        var4.set('あ＋ア')
    elif(show２ == 1):
        var4.set('あ')
    else:
        var4.set('ア')
    showVar2()


#窗口初始化
windows = tkinter.Tk()
windows.title('aiueo')
windows.geometry('400x600')
#文本框一初始化 显示字母组合
var = tkinter.StringVar()
l = tkinter.Label(windows, textvariable = var, font=('Arial, 30'), width=75, height=2)
l.pack()
#文本框二初始化 显示假名
var2 = tkinter.StringVar()
l2 = tkinter.Label(windows, textvariable = var2, font=('Arial, 30'), width=75, height=2)
l2.pack()
#第一个按键 产生新的组合
b = tkinter.Button(windows, text = 'NEW!', font=('Arial, 30'), width = 20, height=3, command=botton1)
b.pack()
#第二个按键 显示答案
b2 = tkinter.Button(windows, text = "Answer?", font=('Arial, 30'), width = 20, height=3, command=botton2)
b2.pack()
#第三个按键 开关答案显示和隐藏
var3 = tkinter.StringVar()
var3.set('Learning')
b3 = tkinter.Button(windows, textvariable = var3, font=('Arial, 30'), width = 20, height=3, command=botton3)
b3.pack()
#第三个按键 开关答案显示和隐藏
var4 = tkinter.StringVar()
var4.set('あ＋ア')
b4 = tkinter.Button(windows, textvariable = var4, font=('Arial, 30'), width = 20, height=3, command=botton4)
b4.pack()
windows.mainloop()





