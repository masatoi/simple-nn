(defpackage simple-nn
  (:use :cl)
  (:nicknames :snn))
(in-package :simple-nn)

;;; Structures

(defstruct layer
  in-dim   ; input dimension
  out-dim  ; output dimension
  w-mat    ; weight matrix
  u-vec    ; input vector
  z-vec    ; output vector
  delta-vec ; differentiation respect to the input of the error function
  activation-func
  activation-func-diff)

(defstruct nn
  n-of-layers
  layer-vec
  learning-rate)

;;; Constructors

(defun make-random-weight (in-dim out-dim)
  (let ((w (make-array (list out-dim in-dim) :element-type 'double-float)))
    (loop for i from 0 to (1- out-dim) do
      (loop for j from 0 to (1- in-dim) do
        ;; initialize between -0.1 and 0.1
        (setf (aref w i j) (- (random 0.2d0) 0.1d0))))
    w))

(defun make-random-layer (in-dim out-dim activation-func activation-func-diff)
  (make-layer :in-dim in-dim
              :out-dim out-dim
              :w-mat (make-random-weight in-dim out-dim)
              :u-vec (make-array out-dim :element-type 'double-float :initial-element 0d0)
              :z-vec (make-array out-dim :element-type 'double-float :initial-element 0d0)
              :delta-vec (make-array out-dim :element-type 'double-float :initial-element 0d0)
              :activation-func activation-func
              :activation-func-diff activation-func-diff))

(defun make-random-nn (dimension-list activation-func-pair-list &optional (learning-rate 0.01d0))
  (labels ((make-layers (product dimension-list activation-func-pair-list)
             (if (< (length dimension-list) 2)
               (nreverse product)
               (make-layers (cons (make-random-layer (car dimension-list) (cadr dimension-list)
                                                     (caar activation-func-pair-list)
                                                     (cadar activation-func-pair-list))
                                  product)
                            (cdr dimension-list) (cdr activation-func-pair-list)))))
    (make-nn :n-of-layers (1- (length dimension-list))
             :layer-vec (apply #'vector (make-layers nil dimension-list activation-func-pair-list))
             :learning-rate learning-rate)))

;;; Activation functions

;; RLF; Rectified Linear Function
(defun RLF (u)
  (if (> u 0d0) u 0d0))

(defun RLF-diff (u)
  (if (>= u 0d0) 1d0 0d0))

;; Identical function
;; Differntial of identity function
(defun one (x)
  (declare (ignore x))
  1d0)

;; Logistic function
(defun logistic (u)
  (/ 1d0 (+ 1d0 (exp (- u)))))

(defun logistic-diff (u)
  (let ((f (logistic u)))
    (* f (- 1d0 f))))

;; Hyperbolic tangent
(declaim (ftype (function (double-float) double-float) tanh-diff))
(defun tanh-diff (u)
  (declare (type double-float u)
           (optimize (speed 3) (safety 0)))
  (let ((tanh-u (tanh u)))
    (- 1d0 (* tanh-u tanh-u))))

;;; Feed-forward

(defun calc-u-vec (in-vec layer)
  (let ((w-mat (layer-w-mat layer)))
    (declare (type (simple-array double-float) in-vec w-mat)
             (optimize (speed 3) (safety 0)))
    (loop for j fixnum from 0 below (layer-out-dim layer) do
      (setf (aref (layer-u-vec layer) j)
            (loop for i fixnum from 0 below (layer-in-dim layer)
                  summing
                  (* (aref w-mat j i)
                     (aref in-vec i))
                  double-float)))
    (layer-u-vec layer)))

(defun calc-z-vec (layer)
  (let ((u-vec (layer-u-vec layer)))
    (declare (type (simple-array double-float) u-vec)
             (optimize (speed 3) (safety 0)))
    (loop for i fixnum from 0 below (layer-out-dim layer) do
      (setf (aref (layer-z-vec layer) i)
            (funcall (layer-activation-func layer) (aref u-vec i))))
    (layer-z-vec layer)))

(defun forward (in-vec nn)
  (loop for i from 0 to (1- (nn-n-of-layers nn)) do
    (if (zerop i)
      (progn (calc-u-vec in-vec (aref (nn-layer-vec nn) i))
             (calc-z-vec (aref (nn-layer-vec nn) i)))
      (progn (calc-u-vec (layer-z-vec (aref (nn-layer-vec nn) (1- i))) (aref (nn-layer-vec nn) i))
             (calc-z-vec (aref (nn-layer-vec nn) i))))))

;;; Back-propagation

(defun calc-last-layer-delta (train-vec last-layer)
  (let ((delta-vec (layer-delta-vec last-layer))
        (z-vec (layer-z-vec last-layer)))
    (declare (type (simple-array double-float) train-vec delta-vec z-vec)
             (optimize (speed 3) (safety 0)))
    (loop for j fixnum from 0 below (layer-out-dim last-layer) do
      (setf (aref delta-vec j)
            (- (aref z-vec j)
               (aref train-vec j))))))

(defun calc-layer-delta (layer next-layer)
  (let ((next-delta-vec (layer-delta-vec next-layer))
        (next-w-mat (layer-w-mat next-layer))
        (u-vec (layer-u-vec layer)))
    (declare (type (simple-array double-float) next-delta-vec next-w-mat u-vec)
             (optimize (speed 3) (safety 0)))
    (loop for j fixnum from 0 below (layer-in-dim next-layer) do
      (setf (aref (layer-delta-vec layer) j)
            (* (funcall (layer-activation-func-diff layer) (aref u-vec j))
               (loop for k fixnum from 0 below (layer-out-dim next-layer)
                     summing
                     (* (aref next-delta-vec k) (aref next-w-mat k j))
                     double-float))))))

(defun backward (train-vec nn)
  ;; calculate last layer's delta
  (let ((last-layer (aref (nn-layer-vec nn) (1- (nn-n-of-layers nn)))))
    (calc-last-layer-delta train-vec last-layer))
  ;; calculate other deltas
  (loop for l from (- (nn-n-of-layers nn) 2) downto 0 do
    (let ((layer (aref (nn-layer-vec nn) l))
          (next-layer (aref (nn-layer-vec nn) (1+ l))))
      (calc-layer-delta layer next-layer))))

(defun predict (in-vec nn)
  (forward in-vec nn)
  (layer-z-vec (aref (nn-layer-vec nn) (1- (nn-n-of-layers nn)))))

(defun update (in-vec train-vec nn)
  (forward in-vec nn)
  (backward train-vec nn)
  ;; update first layer
  (let ((first-layer (aref (nn-layer-vec nn) 0)))
    (loop for i from 0 to (1- (layer-in-dim first-layer)) do
      (loop for j from 0 to (1- (layer-out-dim first-layer)) do
        (setf (aref (layer-w-mat first-layer) j i)
              (- (aref (layer-w-mat first-layer) j i)
                       (* (nn-learning-rate nn)
                          (aref in-vec i)
                          (aref (layer-delta-vec first-layer) j)))))))
  ;; update other layer
  (loop for l from 1 to (1- (nn-n-of-layers nn)) do
    (let ((layer (aref (nn-layer-vec nn) l))
          (prev-layer (aref (nn-layer-vec nn) (1- l))))
      (loop for i from 0 to (1- (layer-in-dim layer)) do
        (loop for j from 0 to (1- (layer-out-dim layer)) do
          (setf (aref (layer-w-mat layer) j i)
                (- (aref (layer-w-mat layer) j i)
                         (* (nn-learning-rate nn)
                            (aref (layer-z-vec prev-layer) i)
                            (aref (layer-delta-vec layer) j)))))))))
