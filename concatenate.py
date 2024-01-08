import glob
import os
import shutil

os.makedirs('test', exist_ok=True)
folders = glob.glob('logs/version_8/test/*')
case = [folders[i].split('/')[-1] for i in range(len(folders))]

for i in range(len(folders)):
    os.makedirs('test/'+case[i], exist_ok=True)
    files = glob.glob(folders[i]+'/*')
    for j in range(len(files)):
        shutil.copy(files[j], 'test/'+case[i]+'/'+files[j].split('/')[-1])

    # m11 m21 m22 m31 m32 m33
    m11 = open('logs/version_8/test/'+case[i]+'/m.sol', 'r')
    m21 = open('logs/version_9/test/'+case[i]+'/m.sol', 'r')
    m22 = open('logs/version_10/test/'+case[i]+'/m.sol', 'r')
    m31 = open('logs/version_12/test/'+case[i]+'/m.sol', 'r')
    m32 = open('logs/version_13/test/'+case[i]+'/m.sol', 'r')
    m33 = open('logs/version_15/test/'+case[i]+'/m.sol', 'r')

    m11_pred = open('logs/version_8/test/'+case[i]+'/m_pred.sol', 'r')
    m21_pred = open('logs/version_9/test/'+case[i]+'/m_pred.sol', 'r')
    m22_pred = open('logs/version_10/test/'+case[i]+'/m_pred.sol', 'r')
    m31_pred = open('logs/version_12/test/'+case[i]+'/m_pred.sol', 'r')
    m32_pred = open('logs/version_13/test/'+case[i]+'/m_pred.sol', 'r')
    m33_pred = open('logs/version_15/test/'+case[i]+'/m_pred.sol', 'r')

    m11 = m11.readlines()
    m21 = m21.readlines()
    m22 = m22.readlines()
    m31 = m31.readlines()
    m32 = m32.readlines()
    m33 = m33.readlines()

    m11_pred = m11_pred.readlines()
    m21_pred = m21_pred.readlines()
    m22_pred = m22_pred.readlines()
    m31_pred = m31_pred.readlines()
    m32_pred = m32_pred.readlines()
    m33_pred = m33_pred.readlines()

    intro = m11[0:6]

    with open('test/'+case[i]+'/m.sol', 'w') as f:
        f.writelines(intro)
        f.write('1 3\n')
        for j in range(7, len(m11)):
            f.write(m11[j][:-1] + ' ' + m21[j][:-1] + ' ' + m22[j][:-1] + ' ' + m31[j][:-1] + ' ' + m32[j][:-1] + ' ' + m33[j][:-1] + '\n')
    with open('test/'+case[i]+'/m_pred.sol', 'w') as f:
        f.writelines(intro)
        f.write('1 3\n')
        for j in range(7, len(m11_pred)):
            f.write(m11_pred[j][:-1] + ' ' + m21_pred[j][:-1] + ' ' + m22_pred[j][:-1] + ' ' + m31_pred[j][:-1] + ' ' + m32_pred[j][:-1] + ' ' + m33_pred[j][:-1] + '\n')