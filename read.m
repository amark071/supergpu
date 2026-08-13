file_name = "data/stiffness_matrix.dat";
f = fopen(file_name);
data = textscan(f, "%d %d %f", "Delimiter", " ", "CollectOutput", true, "HeaderLines", 1);
%data = textscan(f, "%d %d %f", "Delimiter", " ", "CollectOutput", true);
rows = data{1}(:, 1);
cols = data{1}(:, 2);
vals = data{2};
A = sparse(rows, cols, vals);

