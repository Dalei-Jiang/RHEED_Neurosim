# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:05:44 2024

@author: dalei
"""
import time
import torch
import smtplib
import numpy as np
from email.mime.multipart import MIMEMultipart
from torch.utils.data import DataLoader, Dataset
from email.mime.text import MIMEText

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=int)

def email(body, subject):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    # TODO input the sender email
    sender_email = "sjzwyzmydz4869@gmail.com"
    # TODO input password
    sender_password = "wxhw ltlb mzpb yakh"  
    # TODO input receiver email
    recipient_email = "daleij@umich.edu"

    subject = subject
    body = body
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        # 连接到SMTP服务器并发送邮件
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # 开始TLS加密
        server.login(sender_email, sender_password)  # 登录
        server.sendmail(sender_email, recipient_email, msg.as_string())  # 发送邮件
        server.quit()  # 断开连接
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
    print(body)
    
def softmax(x):
    exp_x = np.exp(8*(x - np.max(x, axis=1, keepdims=True)))  # 稳定性调整
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def calculate_final_scores(logits):
    probabilities = softmax(logits)
    scores = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70])
    final_scores = np.sum(probabilities * scores, axis=1)
    return final_scores