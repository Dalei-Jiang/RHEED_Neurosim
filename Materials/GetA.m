clc;clear;close all;
LTP = load('./weight/LTP.mat').LTP;
starting_LTP = min(LTP);
LTP_normal = LTP;
LTP_normal(:,1) = LTP(:,1) - starting_LTP(1,1);
LTP_normal(:,2) = LTP(:,2) - starting_LTP(1,2);
ending_LTP = max(LTP_normal);
max_level = ending_LTP(1,1);
max_current = ending_LTP(1,2);
LTP_normal(:,1) = LTP_normal(:,1)/max_level;
LTP_normal(:,2) = LTP_normal(:,2)/max_current;
FT=fittype('(1-exp(-x/A))/(1-exp(-1/A))');
opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                  'StartPoint', 1, ...
                  'Lower', 0.01, ...   
                  'Upper', 100, ...     
                  'MaxFunEvals', 1000, ... 
                  'MaxIter', 1000, ...   
                  'TolFun', 1e-6, ...   
                  'TolX', 1e-6);
LTP_FT=fit(LTP_normal(:,1),LTP_normal(:,2),FT,opts);
LTP_A = LTP_FT.A;
LTP_fit = (1 - exp(-LTP_normal(:,1) / LTP_A)) / (1 - exp(-1 / LTP_A));

LTD = load('./weight/LTD.mat').LTD;
LTD_normal = LTD;
max_level = max(LTD);
max_current = max_level(1,2);
LTD_normal(:,2) = max_current - LTD_normal(:,2);
min_level = min(LTD);
LTD_normal(:,1) = LTD_normal(:,1) - min_level(1,1);
max_level = max(LTD_normal);
LTD_normal(:,1) = LTD_normal(:,1) / max_level(1,1);
LTD_normal(:,2) = LTD_normal(:,2) / max_level(1,2);
LTD_FT=fit(LTD_normal(:,1),LTD_normal(:,2),FT,opts);
LTD_A = LTD_FT.A;
LTD_fit = (1 - exp(-LTD_normal(:,1) / LTD_A)) / (1 - exp(-1 / LTD_A));

hold on;
title('The fitting of normalized LTP and LTD curve')
plot(LTP_normal(:,1),LTP_normal(:,2),'ro')
plot(LTP_normal(:,1),LTP_fit,'r-')
plot((LTD_normal(:,1)),flipud(1.0-LTD_normal(:,2)),'bo')
plot(LTD_normal(:,1),flipud(1.0-LTD_fit),'b-')
legend('LTP','LTP fitting','LTD','LTD fitting','Location', 'northwest')
hold off;

