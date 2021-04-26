import glob
import re
import os


def search(dirname, keyword):

    try:
        fnames = os.listdir(dirname)
        for fname in fnames:
            full_filename = os.path.join(dirname, fname)

            if os.path.isdir(full_filename):
                search(full_filename, keyword) # Recursive for directory
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext=='.txt' or ext=='.c' or ext=='.h' or ext=='.cpp':
                    # 파일내에서 키워드 찾기
                    find_keyword(full_filename, keyword)

            # 확장자 추출 (.txt .h .c .cpp)
            

            #print(full_filename)
    
    except PermissionError:
        print("permission error occoured")
        pass


def find_keyword(fname, keyword):

    for word in keyword:

        p= re.compile(word)
 
        try:

            with open(fname, "r") as f:
                for x, y in enumerate(f.readlines(), 1):
                    m = p.findall(y)
                    if m:
                        print('File %s [ %d ] Line Searching : %s' %(fname, x, m))
                        print('Full Line Test : %s' %y)
                        cmdstr = "sudo mv " + fname + " /home/minkyung/recover/"
                        os.system(cmdstr)
        except:
            pass



dname = str(input('Input searching directory: '))

#word = ["CMovingBall::", "GzSpinDef.h", "GzSpinUtils.h"]
word = ["masked_adpt_th", "ResizeMarkRegion"]

#word = str(input('Input keyword: '))
search(dname, word)


# s = str(input('Input searching text : '))
# p = re.compile(s)

# for i in glob.glob("./*.txt"):
#     with open(i, 'r') as f:
#         for x, y in enumerate(f.readlines(), 1):
#             m = p.findall(y)
#             if m:
#                 print('File %s [ %d ] Line Searching : %s' %(i,x,m))
#                 print('Full Line Test : %s' %y)

#         print()
