classdef MyApp < matlab.apps.AppBase
    
    % Properties that correspond to app components
    properties (Access = public)
        UIFigure      matlab.ui.Figure
        UIAxes        matlab.ui.control.UIAxes
        UploadButton  matlab.ui.control.Button
        ProcessButton matlab.ui.control.Button
    end
    
    properties (Access = private)
        img % uploaded image 
        mdl % ML model
    end
    
    % Callbacks that handle component events
    methods (Access = private)
        
        % Button pushed function: UploadButton
        function UploadButtonPushed(app,~)
            [filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp','Image Files (*.jpg,*.jpeg,*.png,*.bmp)'});
            if isequal(filename,0) || isequal(pathname,0)
                % User clicked cancel
                return;
            end
            % Load image and display it in the panel
            
            app.img = imread(fullfile(pathname, filename));
            app.UIAxes.Visible = 'on';
            resizedImg = imresize(app.img, ...
                [app.UIAxes.Position(4) app.UIAxes.Position(3)]);
            imshow(resizedImg, 'Parent', app.UIAxes);

            
%             img = imread(fullfile(pathname, filename));
%             app.UIAxes.Visible = 'on';
%             imshow(img, 'Parent', app.UIAxes);
        end
        
        % Button pushed function: ProcessButton
        function ProcessButtonPushed(app,~)
            % Get the image from the panel
            %img = getimage(app.UIAxes);
            
            % Process the image using your custom function
            h = waitbar(0,'Processing, please wait...');
            %processedImg = app.mdl.processImage(img);
            processedImg = app.mdl.processImage(app.img);
            
            % Close waitbar
            close(h); 
            
            % Display the updated image in the panel
            app.UIAxes.Visible = 'on';
            
            %resizedImg = imresize(processedImg, ...
            %    [app.UIAxes.Position(4) app.UIAxes.Position(3)]);
            %imshow(resizedImg, 'Parent', app.UIAxes);
            
            imshow(processedImg, 'Parent', app.UIAxes);
        end
    end
    
    % App initialization and construction
    methods (Access = private)
        
        % Create UIFigure and components
        function createComponents(app)
            % Create UIFigure
            app.UIFigure = uifigure('Name', 'My Image App');
            app.UIFigure.Position = [100 100 640 480];
            
            % Create UIPanel
            %app.UIPanel = uipanel(app.UIFigure);
            %app.UIPanel.Title = 'Image';
            %app.UIPanel.Position = [50 50 400 400];
            
            %app.UIAxes = uiaxes(app.UIPanel);
            app.UIAxes = uiaxes(app.UIFigure);
            app.UIAxes.Visible = 'on';
            app.UIAxes.Position = [80 100 480 360];
            app.UIAxes.Box = 'on';
            
            
            % Create UploadButton
            app.UploadButton = uibutton(app.UIFigure, 'push');
            app.UploadButton.ButtonPushedFcn = createCallbackFcn(app, @UploadButtonPushed, true);
            app.UploadButton.Position = [100 40 100 40];
            %app.UploadButton.Position = [500 350 100 22];
            app.UploadButton.Text = 'Upload Image';
            
            % Create ProcessButton
            app.ProcessButton = uibutton(app.UIFigure, 'push');
            app.ProcessButton.ButtonPushedFcn = createCallbackFcn(app, @ProcessButtonPushed, true);
            app.ProcessButton.Position = [440 40 100 40];
            %app.ProcessButton.Position = [500 300 100 22];
            app.ProcessButton.Text = 'Process Image';
        end
    end
    
    % App creation and deletion
    methods (Access = public)
        
        % Construct app
        function app = MyApp
            
            % Create UIFigure and components
            createComponents(app);
            
            % Load the pre-trained KKLDJ model 
            %app.mdl = load('ModelKKLDJ.mat').mdl;
            %app.mdl = load('ModelKKLDJ_GLCM_HSV.mat').mdl;
            app.mdl = load('ModelKKLDJ_bayesopt_04-24.mat').mdlopt;
            
            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
            app.UIFigure.Name = 'My Image App';
        end
    end
end
