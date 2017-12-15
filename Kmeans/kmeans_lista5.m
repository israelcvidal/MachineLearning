color1 = [228,26,28]/255;
color2 = [55,126,184]/255;
color3 = [77,175,74]/255;
color4 = [152,78,163]/255;
color5 = [255,127,0]/255;
data = dlmread('ex5data1.data');

% Vai retornar idx como a classificacao dos dados nos k clusters
% Vai retornar C como a posicao do centroide
% sumd retorna a soma da distancia pro centroide

%k = 2
[idx,C,sumd] = kmeans(data(:,1:4),2);

% figure;
% gscatter(data(:,1),data(:,2),idx,...
%     [color1;color4],'oo');
% legend('Group 1','Group 2');

Y = sum(sumd);

%k = 3
[idx,C,sumd] = kmeans(data(:,1:4),3);

figure;
gscatter(data(:,1),data(:,2),idx,...
    [color4;color3;color1],'ooo');
legend('Group 1','Group 2','Group 3');

Y = [Y ; sum(sumd)];

%k = 4
[idx,C,sumd] = kmeans(data(:,1:4),4);

% figure;
% gscatter(data(:,1),data(:,2),idx,...
%     [color1;color4;color3;color5],'oooo');
% legend('Group 1','Group 2','Group 3','Group 4');
% 
Y = [Y ; sum(sumd)];

%k = 5
[idx,C,sumd] = kmeans(data(:,1:4),5);

% figure;
% gscatter(data(:,1),data(:,2),idx,...
%     [color1;color4;color3;color5;color2],'ooooo');
% legend('Group 1','Group 2','Group 3','Group 4','Group 5');

Y = [Y ; sum(sumd)];
X = [2,3,4,5];

 figure;
 plot(X,Y, 'Color', color1);

figure;
plot(data(1:50,1),data(1:50,2),'o','MarkerSize',8,'Color',color1);
hold on;
plot(data(51:100,1),data(51:100,2),'o','MarkerSize',8,'Color',color3);
plot(data(101:150,1),data(101:150,2),'o','MarkerSize',8,'Color',color4);
legend('Class 1','Class 2','Class 3');
hold off;
 
idx = kmeans(data,3);
