(in-package :simple-nn)

(defparameter input-data
  (loop repeat 100 collect (vector (- (random (* 2 pi)) pi) 1d0)))

(defparameter train-data
  (mapcar (lambda (x) (vector (sin (aref x 0)))) input-data))

(defparameter *nn*
  (make-random-nn
   '(2 50 1)                       ; 入力層2次元、隠れ層50次元、出力層1次元
   (list (list #'RLF #'RLF-diff)   ; 隠れ層の活性化関数: 正規化線形関数
         (list #'identity #'one))  ; 出力層の活性化関数: 回帰問題なので恒等写像
   0.05d0))

(require 'sb-sprof)
(sb-sprof:with-profiling (:max-samples 1000
			  :report :flat
			  :loop nil)
  (dotimes (i 1000)
    (mapc (lambda (in out) (update in out *nn*)) input-data train-data)))

(time
 (dotimes (i 1000)
   (mapc (lambda (in out) (update in out *nn*)) input-data train-data)))

(defparameter *x*
  (loop for x from -3.14 to 3.14 by 0.01 collect x))

(defparameter *result*
  (mapcar (lambda (x)
            (aref (predict (vector (* x 1d0) 1d0) *nn*) 0))
          *x*))

(defparameter *nn2*
  (make-random-nn
   '(2 50 1)                       ; 入力層2次元、隠れ層50次元、出力層1次元
   (list (list #'tanh #'tanh-diff)   ; 隠れ層の活性化関数: 正規化線形関数
         (list #'identity #'one))  ; 出力層の活性化関数: 回帰問題なので恒等写像
   0.05d0))

(sb-sprof:with-profiling (:max-samples 1000
			  :report :flat
			  :loop nil)
  (dotimes (i 1000)
    (mapc (lambda (in out) (update in out *nn2*)) input-data train-data)))

(time
 (dotimes (i 1000)
   (mapc (lambda (in out) (update in out *nn2*)) input-data train-data)))

;; Evaluation took:
;;   2.098 seconds of real time
;;   2.085005 seconds of total run time (2.080718 user, 0.004287 system)
;;   [ Run times consist of 0.018 seconds GC time, and 2.068 seconds non-GC time. ]
;;   99.38% CPU
;;   7,119,363,056 processor cycles
;;   2,964,792,112 bytes consed

(defparameter *result2*
  (mapcar (lambda (x)
            (aref (predict (vector (* x 1d0) 1d0) *nn2*) 0))
          *x*))

(ql:quickload :clgplot)

(clgp:plots (list *result*
                  *result2*
                  (mapcar #'sin *x*))
            :x-seqs (list *x* *x* *x*))
