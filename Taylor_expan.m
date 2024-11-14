function output = Taylor_expan(x, power)
    [row, col] = size(x);
    output = ones(1, col);
    
    for i = 1:power
        indice = generate_combinations(row, i);
        output = [output; calculate_product(indice, x)];
    end
    
    % 嵌套函数：generate_combinations
    function combinations = generate_combinations(n, m)
        combinations = zeros(m, nchoosek(n+m-1, m));
        counter = 1;
        generate_combinations_helper(zeros(1, m), 1, 1);
        
        function generate_combinations_helper(current_vector, start, depth)
            if depth > m
                combinations(:, counter) = current_vector;
                counter = counter + 1;
                return;
            end
            
            for i = start:n
                current_vector(depth) = i;
                generate_combinations_helper(current_vector, i, depth + 1);
            end
        end
    end
    
    % 嵌套函数：calculate_product
    function X = calculate_product(combinations, x)
        [m, numComb] = size(combinations);
        numCols = size(x, 2);
        x_expanded = reshape(x(combinations, :), [m, numComb, numCols]);
        X = squeeze(prod(x_expanded, 1));
    end
end
