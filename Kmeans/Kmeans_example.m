%%%%%%%%%%%%%%%%%%% 数据生成 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 2000;                % 样本量大小
X = rand(n,2)*10;        % 2n * 2的数据矩阵，每一行表示一个数据点，第一列表示x轴坐标，第二列表示y轴坐标
Y = zeros(n,1);          % 类别标签

for i=1:n
   if 0<X(i,1) && X(i,1)<3 && 0<X(i,2) && X(i,2)<3              % 根据x和y轴坐标确定分类      
       Y(i) = 1;
   end
   if 0<X(i,1) && X(i,1)<3 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 1;
   end
   if 0<X(i,1) && X(i,1)<3 && 7<X(i,2) && X(i,2)<10
       Y(i) = 1;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 0<X(i,2) && X(i,2)<3
       Y(i) = 1;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 1;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 7<X(i,2) && X(i,2)<10
       Y(i) = 1;
   end
   if 7<X(i,1) && X(i,1)<10 && 0<X(i,2) && X(i,2)<3
       Y(i) = 1;
   end
   if 7<X(i,1) && X(i,1)<10 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 1;
   end;
   if 7<X(i,1) && X(i,1)<10 && 7<X(i,2) && X(i,2)<10
       Y(i) = 1;
   end;
end
X = X(Y>0,:);                                                    % 注意X是在[0,10]*[0,10]范围内均匀生成的，而我们只标出了一部分X，类别之间的白色间隔中的点没有标，因此需要将这些点去掉
Y = Y(Y>0,:);                                                    % X(Y>0,:)表示只取X中对应的Y大于0的行，这是因为白色间隔中的点的Y都为0
n = length(Y);                                                   % 去除掉白色间隔剩下的点的个数

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(:,1),X(:,2),'ko','LineWidth',1,'MarkerSize',10);            % 画每类数据点    X(：,1)表示所有点的第一维度坐标
hold on;
xlabel('x axis');
ylabel('y axis');

clear Y;                                                                 % 类别信息仅用与生成数据

%%%%%%%%%%%%%%%%%%  K-means算法：学生实现     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  画出聚类结果，注意类别信息Y是不能使用的     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = 9;                                 % 中心点个数
Ym = zeros(n,1);                       % 每个数据点的预测输出类别
meanpoint = rand(K,2)*10;              % K个初始中心


%%%%%%%%%%%%%%%%%%  画出聚类结果及中心点     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(2)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Ym==1,1),X(Ym==1,2),'ro','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(Ym==2,1),X(Ym==2,2),'ko','LineWidth',1,'MarkerSize',10);            % 画第二类数据点
hold on;
plot(X(Ym==3,1),X(Ym==3,2),'bo','LineWidth',1,'MarkerSize',10);            % 画第三类数据点
hold on;
plot(X(Ym==4,1),X(Ym==4,2),'g*','LineWidth',1,'MarkerSize',10);            % 画第四类数据点
hold on;
plot(X(Ym==5,1),X(Ym==5,2),'m*','LineWidth',1,'MarkerSize',10);            % 画第五类数据点
hold on;
plot(X(Ym==6,1),X(Ym==6,2),'c*','LineWidth',1,'MarkerSize',10);            % 画第六类数据点
hold on;
plot(X(Ym==7,1),X(Ym==7,2),'b+','LineWidth',1,'MarkerSize',10);            % 画第七类数据点
hold on;
plot(X(Ym==8,1),X(Ym==8,2),'r+','LineWidth',1,'MarkerSize',10);            % 画第八类数据点
hold on;
plot(X(Ym==9,1),X(Ym==9,2),'k+','LineWidth',1,'MarkerSize',10);            % 画第九类数据点
hold on;
plot(meanpoint(:,1),meanpoint(:,2),'ms','MarkerFaceColor','m','LineWidth',1,'MarkerSize',10);    % 画出中心点
hold on;
xlabel('x axis');
ylabel('y axis');