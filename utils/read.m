input_file = "E:\mumps\supergpu\data\stiffness_matrix.dat";

f = fopen(input_file, "r");
if f == -1
    error("无法打开输入文件: %s", input_file);
end

data = textscan(f, "%d %d %f", ...
    "Delimiter", " ", ...
    "MultipleDelimsAsOne", true, ...
    "CollectOutput", true, ...
    "HeaderLines", 1);

fclose(f);

rows = double(data{1}(:, 1));
cols = double(data{1}(:, 2));
vals = data{2};

% MATLAB 内部使用 CSC 格式
A = sparse(rows, cols, vals);

figure
spy(A)

% 提取 CSC 数据
[m, n] = size(A);

if m ~= n
    error("矩阵必须是方阵，当前大小为 %d × %d", m, n);
end

[row_indices, col_indices, csc_values] = find(A);
nnz_A = numel(csc_values);

% 统计每列非零元素数量
column_counts = accumarray( ...
    col_indices, ...
    ones(size(col_indices)), ...
    [n, 1]);

% 一基 CSC 列指针
column_pointers = [1; 1 + cumsum(column_counts)];

% 输出 CSC 文件
output_file = "E:\mumps\supergpu\data\stiffness_matrix_csc.dat";

f = fopen(output_file, "w");
if f == -1
    error("无法创建输出文件: %s", output_file);
end

% 第一部分：行数和列数
fprintf(f, "%d %d\n", m, n);

% 第二部分：N+1 个一基列指针
fprintf(f, "%d ", column_pointers);
fprintf(f, "\n");

% 第三部分：nnz 个一基行下标
fprintf(f, "%d ", row_indices);
fprintf(f, "\n");

% 第四部分：nnz 个非零值
fprintf(f, "%.9g ", csc_values);
fprintf(f, "\n");

fclose(f);

fprintf("CSC矩阵已经输出到：%s\n", output_file);
fprintf("矩阵大小：%d × %d\n", m, n);
fprintf("非零元素数量：%d\n", nnz_A);