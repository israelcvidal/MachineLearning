clear all
close all
clc
 
 
# geraусo dos dados
load('ex3data1.mat')
 
 #ParРmetros
Dn=X';
y = T';


[LinD ColD]=size(X');

I=randperm(ColD);
Dn=Dn(:,I);
alvos=y(:,I);   % Embaralha saidas desejadas tambem p/ manter correspondencia com vetor de entrada

# Define tamanho dos conjuntos de treinamento/teste (hold out)
ptrn=0.8;    % Porcentagem usada para treino
ptv=0.1;
ptst=0.1; % Porcentagem usada para teste
 
J=floor(ptrn*ColD);
 
L=floor(ptv*ColD);
 
% Vetores para treinamento e saidas desejadas correspondentes
P = Dn(:,1:J); T1 = alvos(:,1:J); 
[lP cP]=size(P);   % Tamanho da matriz de vetores de treinamento
 
% Vetores para teste e saidas desejadas correspondentes
Q = Dn(:,J+1:J+L); T2 = alvos(:,J+1:J+L); 
[lQ cQ]=size(Q);   % Tamanho da matriz de vetores de teste
 
% Vetores para validaусo e saidas desejadas correspondentes
R = Dn(:,J+L+1:end); T3 = alvos(:,J+L+1:end); 
[lR cR]=size(R);   % Tamanho da matriz de vetores de validaусo
 
 
% DEFINE ARQUITETURA DA REDE
%===========================
Ne = 500; % No. de epocas de treinamento
Nh = 10;   % No. de neuronios na camada oculta
No = 10;   % No. de neuronios na camada de saida
 
alfa=0.05;   % Passo de aprendizagem
 
% Inicia matrizes de pesos
WW=0.1*rand(Nh,lP+1);   % Pesos entrada -> camada oculta
 
MM=0.1*rand(No,Nh+1);   % Pesos camada oculta -> camada de saida
 
E = zeros(Ne,1);
 
Ev = zeros(Ne,1);
 
Et = 0;
 
 bool = true;
 t = 1;
%%% ETAPA DE TREINAMENTO
while bool
 
    I=randperm(cP); P=P(:,I); T1=T1(:,I);   % Embaralha vetores de treinamento e saidas desejadas
 
    EQ=0;
    for tt=1:cP,   % Inicia LOOP de epocas de treinamento
        % CAMADA OCULTA
        X=[-1; P(:,tt)];      % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
        Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)
 
        % CAMADA DE SAIDA 
        Y=[-1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
        Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)
 
        % CALCULO DO ERRO 
        Ek = T1(:,tt) - Ok;           % erro entre a saida desejada e a saida da rede
         
        E(t) += sum(Ek.^2);
 
        %%% CALCULO DOS GRADIENTES LOCAIS
        Dk = Ok.*(1 - Ok);  % derivada da sigmoide logistica (camada de saida)
        DDk = Ek.*Dk;       % gradiente local (camada de saida)
 
        Di = Yi.*(1 - Yi); % derivada da sigmoide logistica (camada oculta)
        DDi = Di.*(MM(:,2:end)'*DDk);    % gradiente local (camada oculta)
 
        % AJUSTE DOS PESOS - CAMADA DE SAIDA
        MM = MM + alfa*DDk*Y';
 
        % AJUSTE DOS PESOS - CAMADA OCULTA
        WW = WW + alfa*DDi*X';
        % MEDIA DO ERRO QUADRATICO P/ EPOCA
    end
     
    E(t) /= cP;
     
    %% ETAPA DE VALIDAК├O  %%%
    EQ2=0;
    HID2=[];
    OUT2=[];
    for tt=1:cR,
        % CAMADA OCULTA
        X=[-1; R(:,tt)];      % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
        Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)
         
        % CAMADA DE SAIDA 
        Y=[-1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
        Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)
        OUT2=[OUT2 Ok];       % Armazena saida da rede
         
        Ek = T3(:,tt) - Ok;           % erro entre a saida desejada e a saida da rede
         
        Ev(t) += sum(Ek.^2);
         
         
    end
 
    
    Ev(t) /= cR;
    if t > 1 && Ev(t) > Ev(t-1)
      bool = false;
    endif
    t = t+1;
end   % Fim do loop de treinamento
 
%% ETAPA DE TESTE  %%%
EQ2=0;
HID2=[];
OUT2=[];
for tt=1:cQ,
    % CAMADA OCULTA
    X=[-1; Q(:,tt)];      % Constroi vetor de entrada com adicao da entrada x0=-1
    Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
    Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)
     
    % CAMADA DE SAIDA 
    Y=[-1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
    Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
    Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)
    OUT2=[OUT2 Ok];       % Armazena saida da rede
     
    Ek = T2(:,tt) - Ok;           % erro entre a saida desejada e a saida da rede
     
    Et += sum(Ek.^2);
     
end
 
Et /= cQ
 
 
% CALCULA TAXA DE ACERTO
count_OK=0;  % Contador de acertos
for i=1:cQ,
    [T2max iT2max]=max(T2(:,i));  % Indice da saida desejada de maior valor
    [OUT2_max iOUT2_max]=max(OUT2(:,i)); % Indice do neuronio cuja saida eh a maior
    if iT2max==iOUT2_max,   % Conta acerto se os dois indices coincidem 
    count_OK=count_OK+1;
    end
end
 