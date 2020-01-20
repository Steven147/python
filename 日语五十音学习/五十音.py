import random
import tkinter

#initialize
v1 = '@'
v2 = '@'
alphabet1 = "akstnhmyrw"
values1 = range(0,10,1)
alphabet2 = "aiueo"
values2 = range(0,5,1)
mystr = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもや0ゆ0よらりるれろわ000を"
mystr2 = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤ0ユ0ヨラリルレロワ000ヲ"
show = False

#窗口初始化
windows = tkinter.Tk()
windows.title('五十音学习')
windows.geometry('150x190')

#文本框一初始化 显示字母组合
var = tkinter.StringVar()
l = tkinter.Label(windows, textvariable = var, font=('Arial, 12'), width=15, height=2)
l.pack()

#文本框二初始化 显示假名
var2 = tkinter.StringVar()
l2 = tkinter.Label(windows, textvariable = var2, font=('Arial, 12'), width=15, height=2)
l2.pack()

#按键一对应函数
def new():
    global v1, v2,values1, values2, show
    valid = False
    
    #得到一个有效的字母组合
    while not valid:
        v1 = random.choice(values1)
        v2 = random.choice(values2)
        #71yi 73ye 91wi 92wu 93we
        if(v1*5+v2 == 36 or v1*5+v2 == 38 or v1*5+v2 == 46 or v1*5+v2 == 47 or v1*5+v2 == 48): 
            pass
        else:
            valid = True
            
    #输出字母组合 排除特定情况
    if(v1==0):
        var.set(alphabet2[v2])
    elif(v1*5+v2==16):
        var.set('ch'+alphabet2[v2])
    elif(v1*5+v2==17):
        var.set('ts'+alphabet2[v2])
    else:
        var.set(alphabet1[v1]+alphabet2[v2])
    
    #是否输出假名答案
    if(not show): 
        var2.set('')
    else:
        var2.set(mystr[v1*5+v2]+' '+mystr2[v1*5+v2])
#第一个按键 产生新的组合
b = tkinter.Button(windows, text = 'NEW!', width = 15, height=2, command=new)
b.pack()


#按键二对应函数
def showResult():
    global v1, v2, mystr
    var2.set(mystr[v1*5+v2]+' '+mystr2[v1*5+v2])
#第二个按键 显示答案
b2 = tkinter.Button(windows, text = "I don't know!", width = 15, height=2, command=showResult)
b2.pack()


#按键三对应函数
def alwaysShow():
    global show
    if(not show): 
        var3.set("Don't Show")
        var2.set(mystr[v1*5+v2]+' '+mystr2[v1*5+v2])
        show = not show
    else:
        var3.set('Always Show')
        var2.set('')
        show = not show

#第三个按键 开关答案显示和隐藏
var3 = tkinter.StringVar()
var3.set('Always Show')
b3 = tkinter.Button(windows, textvariable = var3, width = 15, height=2, command=alwaysShow)
b3.pack()
windows.mainloop()




