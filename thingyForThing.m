% --- COMPLETE HARD RESET SCRIPT ---
clear all; close all; clc;

%% 1. Calculate Arc Length & Targets
speed_fun = @(t) sqrt(5*t.^2 + 2*t + 1) ./ (t + 1).^3;
total_length = quadgk(speed_fun, 0, Inf);

N = 20;
t_values = zeros(N, 1);
s_targets = linspace(total_length/N, total_length, N);

%% 2. Solve for t
options = optimset('Display', 'off'); 
current_guess = 0.01; 

for i = 1:(N-1)
    obj_fun = @(t) quadgk(speed_fun, 0, t) - s_targets(i);
    t_values(i) = fzero(obj_fun, current_guess, options);
    current_guess = t_values(i) + 0.1; 
end
t_values(N) = 10000; % Cap the final point to avoid Infinity errors

%% 3. Generate Coordinates
x_coords = t_values ./ (t_values + 1);
y_coords = (t_values ./ (t_values + 1)).^2;

%% 4. Plotting (Bulletproof Vector Method)
figure;

% Draw a faint gray line for the actual path, followed by solid red dots
plot(x_coords, y_coords, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5);
hold on;
plot(x_coords, y_coords, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
hold off;

% Force the axis boundaries so the curve isn't cramped
axis([0 1.05 0 1.05]);
grid on; 
box on; % Forces the solid black border around the plot


%% 5. Export to SVG
set(gcf, 'PaperPositionMode', 'auto');

% 'drawnow' forces Octave to finish visually building the plot before exporting
drawnow; 

print('arc_length_points.svg', '-dsvg');
disp('Script complete. Check your folder for the new SVG.');

