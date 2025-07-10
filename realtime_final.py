import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import os
import mediapipe.python.solutions.drawing_utils as drawing_utils
import mediapipe.python.solutions.drawing_styles as drawing_styles
import time
import re
import joblib # Importar para carregar o modelo ML
# from sklearn.preprocessing import StandardScaler # Se você usou StandardScaler no treinamento, precisaria carregá-lo aqui também

class RealtimeMovementValidatorML: 
    def __init__(self, model_path, ref_csv_paths, min_angle_ref_state, max_angle_ref_state, timeout_duration_seconds=5):
        """
        Inicializa o validador de movimento em tempo real com um modelo de ML.

        Args:
            model_path (str): Caminho para o arquivo .joblib do modelo ML treinado.
            ref_csv_paths (list): Lista de caminhos para os arquivos CSV de referência (usados para carregar, mas os limiares de validação ML vêm do modelo).
            min_angle_ref_state (float): Ângulo mínimo para a máquina de estados detectar o fundo da rosca.
            max_angle_ref_state (float): Ângulo máximo para a máquina de estados detectar a extensão do braço.
            timeout_duration_seconds (int): Duração em segundos para considerar uma repetição como 'presa' ou incompleta.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.ml_model = joblib.load(model_path) # Carrega o modelo ML treinado
        # self.scaler = joblib.load('scaler.joblib') # Se você salvou um scaler no treinamento, carregue-o aqui

        # Carrega os dados de referência (mantido, mas não usado diretamente para regras)
        self.reference_profiles = self._load_and_process_references(ref_csv_paths)
        
        # Limiares de estado para a máquina de estados (NÃO são os limiares de validação do ML)
        # O modelo ML irá inferir a correção, mas a máquina de estados precisa saber o que é um ciclo.
        self.MIN_ANGLE_THRESHOLD_STATE = min_angle_ref_state      
        self.MAX_ANGLE_THRESHOLD_STATE = max_angle_ref_state      

        # Variáveis de estado para detecção de repetições
        self.repetition_state = "EXTENDED" 
        self.current_repetition_frames = [] 
        self.repetition_count = 0 

        # Variáveis para o feedback pós-análise e timeout
        self.all_repetition_results = [] 
        self.last_state_transition_time = time.time() # Tempo da última transição de estado/início de repetição
        self.timeout_duration_seconds = timeout_duration_seconds # Duração do timeout (e.g., 5 segundos)

    def _load_and_process_references(self, csv_paths):
        """Carrega os CSVs de referência e concatena os dados."""
        all_ref_data = []
        for path in csv_paths:
            df = pd.read_csv(path)
            all_ref_data.append(df)
        return pd.concat(all_ref_data, ignore_index=True)

    def calculate_angle(self, a, b, c):
        """Calcula o ângulo em graus entre três pontos."""
        a = np.array(a) 
        b = np.array(b) 
        c = np.array(c) 
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def _extract_features_for_prediction(self, repetition_frames):
        """
        Extrai as mesmas características resumidas usadas no treinamento do modelo ML
        a partir dos dados brutos de uma repetição.
        """
        # --- NOVO: Verificação para repetições muito curtas ou vazias ---
        if not repetition_frames or len(repetition_frames) < 5: # Mínimo de 5 frames para extração significativa
            print(f"DEBUG: Repetição muito curta (< 5 frames, {len(repetition_frames)} frames) ou vazia para extração de características.")
            return pd.DataFrame() # Retorna DataFrame vazio para indicar que não é válido para ML
        # --- FIM NOVO ---

        temp_df = pd.DataFrame(repetition_frames)

        # Assegurar que as colunas existam para evitar erros se a detecção de pose falhar consistentemente.
        # Caso 'desvio_5_frames' não seja calculado em tempo real e não esteja em current_frame_data, ele será 0.0
        feature_data = {
            'min_angulo': temp_df['angulo'].min(),
            'max_angulo': temp_df['angulo'].max(),
            'mean_angulo': temp_df['angulo'].mean(),
            'std_angulo': temp_df['angulo'].std(),
            
            'min_velocidade': temp_df['velocidade'].min(),
            'max_velocidade': temp_df['velocidade'].max(),
            'mean_velocidade': temp_df['velocidade'].mean(),
            'std_velocidade': temp_df['velocidade'].std(),
            
            'range_ombro_y': temp_df['ombro_y'].max() - temp_df['ombro_y'].min(),
            'mean_ombro_y': temp_df['ombro_y'].mean(),
            'range_cotovelo_y': temp_df['cotovelo_y'].max() - temp_df['cotovelo_y'].min(),
            'mean_cotovelo_y': temp_df['cotovelo_y'].mean(),
            'range_punho_y': temp_df['punho_y'].max() - temp_df['punho_y'].min(),
            'mean_punho_y': temp_df['punho_y'].mean(),
            
            'mean_desvio_5_frames': temp_df['desvio_5_frames'].mean() if 'desvio_5_frames' in temp_df.columns else 0.0, # Pode não estar presente nos frames brutos
            'max_desvio_5_frames': temp_df['desvio_5_frames'].max() if 'desvio_5_frames' in temp_df.columns else 0.0,
            
            'duration_frames': len(repetition_frames)
        }
        
        # Converte para DataFrame de 1 linha e lida com NaNs (std de 1 elemento, etc.)
        features_df = pd.DataFrame([feature_data]).fillna(0)
        
        # IMPORTANTE: Garanta que as colunas (e a ordem) sejam as mesmas do X_train usado no treinamento!
        # Você pode obter a lista de colunas de X_train do seu script de treinamento e defini-la aqui.
        # Exemplo: expected_feature_columns_order = ['min_angulo', 'max_angulo', ...]
        # if set(features_df.columns) != set(expected_feature_columns_order):
        #     print("AVISO: Colunas de características não correspondem às esperadas pelo modelo ML!")
        #     # Lógica para reordenar ou adicionar/remover colunas, se necessário.
        # features_df = features_df[expected_feature_columns_order]
        
        return features_df


    def validate_video_realtime(self, video_path):
        """
        Analisa um vídeo em tempo real e fornece feedback sobre a correção do movimento usando ML.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir o vídeo em {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) 
        prev_angle_for_velocity = None
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            max_height = 500
            scale_factor = max_height / image.shape[0]
            image = cv2.resize(image, (int(image.shape[1] * scale_factor), max_height))
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            feedback_text_display = "Aguardando movimento..."
            feedback_color = (255, 255, 255) # White

            current_time = time.time() 

            if results.pose_landmarks:
                shoulder_lm = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                elbow_lm = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                wrist_lm = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

                shoulder = [shoulder_lm.x, shoulder_lm.y]
                elbow = [elbow_lm.x, elbow_lm.y]
                wrist = [wrist_lm.x, wrist_lm.y]
                
                h, w, _ = image.shape
                shoulder_y_pixel = int(shoulder_lm.y * h)
                elbow_y_pixel = int(elbow_lm.y * h)
                wrist_y_pixel = int(wrist_lm.y * h)

                current_angle = self.calculate_angle(shoulder, elbow, wrist)
                
                current_velocity = 0.0
                if prev_angle_for_velocity is not None:
                    time_diff = 1 / fps
                    current_velocity = (current_angle - prev_angle_for_velocity) / time_diff
                prev_angle_for_velocity = current_angle

                current_frame_data = {
                    'angulo': current_angle,
                    'velocidade': current_velocity,
                    'ombro_y': shoulder_y_pixel,
                    'cotovelo_y': elbow_y_pixel,
                    'punho_y': wrist_y_pixel,
                    # Adicionado 'desvio_5_frames' aqui. No sistema real, você precisaria calcular este desvio
                    # em relação a uma referência, ou preenchê-lo com 0 se não for usado.
                    'desvio_5_frames': 0.0 
                }
                
                
                repetition_completed = False
                
                if self.repetition_state == "EXTENDED":
                    if current_angle < (self.MAX_ANGLE_THRESHOLD_STATE - 10):
                        self.repetition_state = "CURLING_UP"
                        self.last_state_transition_time = current_time
                        # NOVO: Inicia a coleta de quadros da repetição *aqui*
                        self.current_repetition_frames = [current_frame_data] 
                    # else: # Se não entrou em CURLING_UP, continua limpando se não for o inicio de nada
                    #    self.current_repetition_frames = [] # Evita acumular frames antes de uma rep real
                elif self.repetition_state == "CURLING_UP":
                    self.current_repetition_frames.append(current_frame_data) # Continua adicionando
                    if (current_time - self.last_state_transition_time) > self.timeout_duration_seconds:
                        self.repetition_count += 1 
                        rep_is_correct = False
                        rep_feedback = f"Repetição {self.repetition_count} (Incompleta): Movimento interrompido ou muito lento!"
                        repetition_completed = True 
                    elif current_angle < (self.MIN_ANGLE_THRESHOLD_STATE + 10):
                        self.repetition_state = "CURLING_DOWN"
                        self.last_state_transition_time = current_time
                elif self.repetition_state == "CURLING_DOWN":
                    self.current_repetition_frames.append(current_frame_data) # Continua adicionando
                    if (current_time - self.last_state_transition_time) > self.timeout_duration_seconds:
                        self.repetition_count += 1 
                        rep_is_correct = False
                        rep_feedback = f"Repetição {self.repetition_count} (Incompleta): Não retornou à extensão!"
                        repetition_completed = True 
                    elif current_angle > (self.MAX_ANGLE_THRESHOLD_STATE - 10):
                        self.repetition_count += 1 
                        
                        # --- SUBSTITUIÇÃO DA VALIDAÇÃO POR REGRAS PELO MODELO ML ---
                        features_for_prediction = self._extract_features_for_prediction(self.current_repetition_frames)
                        
                        if not features_for_prediction.empty:
                            prediction_encoded = self.ml_model.predict(features_for_prediction)[0]
                            prediction_proba = self.ml_model.predict_proba(features_for_prediction)[0] 
                            
                            rep_is_correct = (prediction_encoded == 1) 
                            
                            if rep_is_correct:
                                rep_feedback = f"Repetição {self.repetition_count}: CORRETO! (Confiança: {prediction_proba[1]:.2f})"
                            else:
                                rep_feedback = f"Repetição {self.repetition_count}: INCORRETO (Confiança: {prediction_proba[0]:.2f})"
                        else:
                            rep_is_correct = False
                            rep_feedback = f"Repetição {self.repetition_count}: Repetição muito curta para análise ML!" 
                        # --- FIM DA SUBSTITUIÇÃO ---

                        repetition_completed = True 

                if repetition_completed:
                    self.all_repetition_results.append({
                        'rep_num': self.repetition_count,
                        'is_correct': rep_is_correct,
                        'reason': rep_feedback.replace(f"Repetição {self.repetition_count}: ", "").replace("(Incompleta): ", "") 
                    })
                    feedback_text_display = rep_feedback
                    feedback_color = (0, 255, 0) if rep_is_correct else (0, 0, 255)
                    
                    self.repetition_state = "EXTENDED" 
                    self.current_repetition_frames = [] # Limpa a lista para a próxima repetição
                    self.last_state_transition_time = current_time 

                if self.repetition_count == 0 and not repetition_completed:
                     feedback_text_display = "Inicie a primeira repetição..."
                     feedback_color = (255, 255, 0) 
                elif not repetition_completed and (self.repetition_state == "CURLING_UP" or self.repetition_state == "CURLING_DOWN"):
                     feedback_text_display = f"Repetição {self.repetition_count + 1}: Em andamento..."
                     feedback_color = (0, 165, 255) 
                
            else: 
                feedback_text_display = "Pose não detectada! Ajuste a câmera."
                feedback_color = (0, 0, 255) 
                self.repetition_state = "EXTENDED" 
                self.current_repetition_frames = [] # Limpa e reseta o timer se a pose sumir
                self.last_state_transition_time = current_time 

            cv2.putText(image, feedback_text_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2)
            cv2.putText(image, f"Repetições: {self.repetition_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2) 
            cv2.imshow('Analise de Movimento ML em Tempo Real', image) 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release() 
        cv2.destroyAllWindows()
        
        self.generate_summary_report() 

    def generate_summary_report(self):
        print("\n--- Relatório de Análise da Série (Baseado em ML) ---")
        
        if not self.all_repetition_results:
            print("Nenhuma repetição foi detectada ou validada.")
            return

        total_reps = len(self.all_repetition_results)
        correct_reps = sum(1 for rep in self.all_repetition_results if rep['is_correct'])
        incorrect_reps = total_reps - correct_reps

        print(f"Total de Repetições Detectadas: {total_reps}")
        print(f"Repetições Corretas: {correct_reps}")
        print(f"Repetições Incorretas: {incorrect_reps}")

        if incorrect_reps > 0:
            print("\nMotivos das Repetições Incorretas:")
            reasons_counts = {}
            for rep in self.all_repetition_results:
                if not rep['is_correct']:
                    reason = rep['reason']
                    reason_clean = re.sub(r'\(Confiança:.*?\)', '', reason) 
                    reason_clean = re.sub(r'\(Min:.*?\)', '', reason_clean)
                    reason_clean = re.sub(r'\(Max:.*?\)', '', reason_clean)
                    reason_clean = re.sub(r'\(Movimento:.*?\)', '', reason_clean)
                    reason_clean = re.sub(r'\(Desvio Vel\.:.*?\)', '', reason_clean)
                    reason_clean = reason_clean.strip().split(',')[0].strip() 
                    
                    reasons_counts[reason_clean] = reasons_counts.get(reason_clean, 0) + 1
            
            for reason, count in reasons_counts.items():
                print(f"- {reason}: {count} vezes")

        print("\n--- Sugestões de Melhoria Geral ---")
        if incorrect_reps == 0 and total_reps > 0:
            print("Parabéns! Todas as repetições foram classificadas como corretas pelo modelo ML.")
        elif incorrect_reps > 0:
            print("- O modelo ML identificou inconsistências. Tente focar em:")
            print("  - Manter a amplitude de movimento completa (extensão e flexão).")
            print("  - Executar o movimento de forma suave e controlada, evitando brusquidão.")
            print("  - Manter a estabilidade do tronco e do ombro durante a execução.")
            print("  - Garantir que cada repetição seja completa e não interrompida.")
            print("  - Rever a qualidade dos seus dados de treinamento, se necessário.")


if __name__ == "__main__":
    trained_model_file = 'rosca_quality_classifier_model.joblib' 
    reference_csv_paths_for_init = ['rosca.csv', 'rosca2.csv'] 
    
    min_angle_for_state_machine = 60    
    max_angle_for_state_machine = 140   

    timeout_seconds_for_state = 3 

    validator_ml = RealtimeMovementValidatorML(
        model_path=trained_model_file,
        ref_csv_paths=reference_csv_paths_for_init,
        min_angle_ref_state=min_angle_for_state_machine,
        max_angle_ref_state=max_angle_for_state_machine,
        timeout_duration_seconds=timeout_seconds_for_state
    )
    
    video_to_analyze_path = 'roscacor.mp4' 

    try:
        validator_ml.validate_video_realtime(video_to_analyze_path)
    except FileNotFoundError:
        print(f"Erro: O modelo '{trained_model_file}' ou os CSVs de referência não foram encontrados.")
        print("Certifique-se de que o script de treinamento foi executado e o modelo foi salvo, e que 'rosca.csv' e 'rosca2.csv' estão no diretório correto.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        print("Verifique se todas as bibliotecas necessárias (opencv-python, numpy, pandas, mediapipe, scikit-learn, joblib) estão instaladas.")