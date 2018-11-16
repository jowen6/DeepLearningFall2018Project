clc
clear

filename = 'sr_True_lr_0.001.csv';
vec = importdata(filename);

filename = 'sr_True_lr_0.005.csv';
vec_2 = importdata(filename);

filename = 'sr_True_lr_0.01.csv';
vec_3 = importdata(filename);

filename = 'sr_True_lr_0.05.csv';
vec_4 = importdata(filename);

xlabel('Training data set size M')
ylabel('Success rate %')
ax = gca;
ax.FontSize = 12;
ytickformat(ax, 'percentage');

hold on
grid on
grid minor
xticks(0:20:300)
axis([0,300,50,100])

plot(vec(:,1), 100*vec(:,2),'o-')
plot(vec_2(:,1), 100*vec_2(:,2),'d-')
plot(vec_3(:,1), 100*vec_3(:,2),'s-')
plot(vec_4(:,1), 100*vec_4(:,2),'<-')

legend('Enriched test set \lambda=0.001','Enriched test set \lambda=0.005', ...
'Enriched test set \lambda=0.01','Enriched test set \lambda=0.05',...
'Location','southeast')


%% ===================================================================
filename = 'cr_True_lr_0.001.csv';
vec = importdata(filename);

filename = 'cr_True_lr_0.005.csv';
vec_2 = importdata(filename);
% 
filename = 'cr_True_lr_0.01.csv';
vec_3 = importdata(filename);

filename = 'cr_True_lr_0.05.csv';
vec_4 = importdata(filename);

figure
grid on
grid minor
xticks(0:20:300)
axis([0,300,50,100])
hold on

xlabel('Training data set size M')
ylabel('Confidence rate %')
ax = gca;
ax.FontSize = 12;
ytickformat(ax, 'percentage');

plot(vec(:,1), 100*(1-vec(:,2)),'o-' )
% plot(vec(:,1), 100*(1-vec(:,3)),'d-' )
plot(vec_2(:,1), 100*(1-vec_2(:,2)),'<-' )
% plot(vec_2(:,1), 100*(1-vec_2(:,3)),'>-' )

plot(vec_3(:,1), 100*(1-vec_3(:,2)),'>-' )

plot(vec_4(:,1), 100*(1-vec_4(:,2)),'d-' )

legend('Enriched test set \lambda=0.001','Enriched test set \lambda=0.005', ...
'Enriched test set \lambda=0.01','Enriched test set \lambda=0.05',...
'Location','southeast')

