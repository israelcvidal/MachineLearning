data = dlmread('ex5data1.data');
data = data(:,1:4);

%Variancia em explained
[coeff,score,latent,tsquared,explained,mu] = pca(data);

%Reduzindo a dimensao para 2
[coeff2,score2,latent2,tsquared2] = pca(data,'NumComponents',2);


%falta plotar o grafico
plot(score2(1:50,1),score2(1:50,2),'x',score2(51:100,1),score2(51:100,2),'*',score2(101:150,1),score2(101:150,2),'o');

