clc;clear;
file_name = "E:\mumps\supergpu\data\A_615069.dat";

f = fopen(file_name, "r");
if f == -1
    error("无法打开 CSC 文件: %s", file_name);
end

try
    % 读取矩阵大小
    matrix_size = fscanf(f, "%d", 2);

    if numel(matrix_size) ~= 2
        error("无法读取矩阵行数和列数");
    end

    row_count = matrix_size(1);
    col_count = matrix_size(2);

    if row_count ~= col_count
        error("输入矩阵不是方阵");
    end

    N = col_count;

    % 读取 N+1 个一基 CSC 列指针
    column_pointers = fscanf(f, "%d", N + 1);

    if numel(column_pointers) ~= N + 1
        error("CSC列指针数量不正确");
    end

    if column_pointers(1) ~= 1
        error("CSC列指针必须从1开始");
    end

    nnz_A = column_pointers(end) - 1;

    if nnz_A < 0
        error("CSC列指针中的非零元素数量无效");
    end

    % 读取一基行下标
    row_indices = fscanf(f, "%d", nnz_A);

    if numel(row_indices) ~= nnz_A
        error("CSC行下标数量不正确");
    end

    % 读取非零元素
    values = fscanf(f, "%f", nnz_A);

    if numel(values) ~= nnz_A
        error("CSC非零元素数量不正确");
    end

    fclose(f);

catch exception
    fclose(f);
    rethrow(exception);
end

% 根据列指针构造每个非零元素的列下标
column_indices = zeros(nnz_A, 1);

for col = 1:N
    first_position = column_pointers(col);
    last_position  = column_pointers(col + 1) - 1;

    if first_position <= last_position
        column_indices(first_position:last_position) = col;
    end
end

% 还原 MATLAB 稀疏矩阵
A = sparse( ...
    double(row_indices), ...
    column_indices, ...
    values, ...
    row_count, ...
    col_count);

fprintf("CSC矩阵读取成功\n");
fprintf("矩阵大小：%d × %d\n", row_count, col_count);
fprintf("非零元素数量：%d\n", nnz_A);
