clc;clear;
%% 矩阵读取
matrix = load('E:/mumps/supernodal_gpu/data/A_1215.dat');

% 矩阵处理，转化为sparse类型
row = matrix(1,1);
col = matrix(2,1);
col_temp = matrix(3:2+col,1);
row_total = matrix(3+col,1);
row_indices = matrix(4+col:4+col+row_total-2,1);
[n,~]=size(matrix);
value = matrix(4+col+row_total-1:n);
col_indices=zeros( matrix(3+col,1)-1,1);
for i=1:col-1
    for j=col_temp(i):col_temp(i+1)
        col_indices(j)=i;
    end
end
for j=col_temp(col): matrix(3+col,1)-1
    col_indices(j)=col;
end
S1 = sparse(row_indices, col_indices, value);

% 矩阵大小读取
[m, n] = size(S1);

% 矩阵分块后进行形状绘制
figure
spy(S1)
r = rank(full(S1));
disp(r);