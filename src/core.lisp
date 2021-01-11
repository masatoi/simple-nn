(defpackage simple-nn
  (:use :cl)
  (:nicknames :snn))
(in-package :simple-nn)

;;; Structures

(defstruct layer
  ;; input dimension
  (in-dim 1 :type fixnum :read-only t)
  ;; output dimension
  (out-dim 1 :type fixnum :read-only t)
  ;; weight matrix
  (w-mat nil :type (simple-array single-float))
  ;; input vector
  (u-vec nil :type (simple-array single-float))
  ;; output vector
  (z-vec nil :type (simple-array single-float))
  ;; differentiation respect to the input of the error function
  (delta-vec nil :type (simple-array single-float))
  ;; activation function
  (activation-func nil :type function :read-only t)
  ;; activation function for calc the differentiation
  (activation-func-diff nil :type function :read-only t))

(defstruct nn
  (n-layers 1 :type fixnum :read-only t)
  (layer-vec nil :type simple-array)
  (learning-rate 0.1 :type single-float))

;;; Constructors

(defun make-random-weight-matrix (in-dim out-dim)
  (let ((w (make-array (list out-dim in-dim) :element-type 'single-float)))
    (loop for i fixnum from 0 below out-dim do
      (loop for j fixnum from 0 below in-dim do
        ;; initialize between -0.1 and 0.1
        (setf (aref w i j) (- (random 0.2) 0.1))))
    w))

(defun make-random-layer (in-dim out-dim activation-func activation-func-diff)
  (declare (type fixnum in-dim out-dim)
           (type (function (single-float) single-float) activation-func activation-func-diff))
  (make-layer :in-dim in-dim
              :out-dim out-dim
              :w-mat (make-random-weight-matrix in-dim out-dim)
              :u-vec (make-array out-dim :element-type 'single-float :initial-element 0.0)
              :z-vec (make-array out-dim :element-type 'single-float :initial-element 0.0)
              :delta-vec (make-array out-dim :element-type 'single-float :initial-element 0.0)
              :activation-func activation-func
              :activation-func-diff activation-func-diff))

(defun make-random-nn (dimension-list activation-func-pair-list &optional (learning-rate 0.01))
  (labels ((make-layers (product dimension-list activation-func-pair-list)
             (if (< (length dimension-list) 2)
                 (nreverse product)
                 (make-layers (cons (make-random-layer (car dimension-list) (cadr dimension-list)
                                                       (caar activation-func-pair-list)
                                                       (cadar activation-func-pair-list))
                                    product)
                              (cdr dimension-list) (cdr activation-func-pair-list)))))
    (make-nn :n-layers (1- (length dimension-list))
             :layer-vec (apply #'vector (make-layers nil dimension-list activation-func-pair-list))
             :learning-rate learning-rate)))

;;; Activation functions

;; RLF; Rectified Linear Function
(declaim (ftype (function (single-float) single-float) RLF RLF-diff))
(defun RLF (u)
  (declare (optimize (speed 3) (safety 0))
           (type single-float u))
  (if (> u 0.0) u 0.0))

(defun RLF-diff (u)
  (declare (optimize (speed 3) (safety 0))
           (type single-float u))
  (if (>= u 0.0) 1.0 0.0))

;; Identical function
;; Differntial of identity function
(declaim (ftype (function (single-float) single-float) one))
(defun one (x)
  (declare (ignore x))
  1.0)

;; Logistic function
(declaim (ftype (function (single-float) single-float) logistic logistic-diff))
(defun logistic (u)
  (declare (optimize (speed 3) (safety 0))
           (type single-float u))
  (/ 1.0 (+ 1.0 (exp (- u)))))

(defun logistic-diff (u)
  (declare (optimize (speed 3) (safety 0))
           (type single-float u))
  (let ((f (logistic u)))
    (* f (- 1.0 f))))

;; Hyperbolic tangent
(declaim (ftype (function (single-float) single-float) tanh-diff))
(defun tanh-diff (u)
  (declare (optimize (speed 3) (safety 0))
           (type single-float u))
  (let ((tanh-u (the single-float (tanh u))))
    (- 1.0 (* tanh-u tanh-u))))

;;; Feed-forward

(defun calc-u-vec (in-vec layer)
  (let ((w-mat (layer-w-mat layer))
        (u-vec (layer-u-vec layer)))
    (declare (optimize (speed 3) (safety 0))
             (type (simple-array single-float) in-vec w-mat u-vec))
    (loop for j fixnum from 0 below (length u-vec) do
      (setf (aref u-vec j)
            (loop for i fixnum from 0 below (length in-vec)
                  summing (* (aref w-mat j i)
                             (aref in-vec i))
                  single-float)))))

(defun calc-z-vec (layer)
  (let ((u-vec (layer-u-vec layer))
        (z-vec (layer-z-vec layer))
        (activation-func (layer-activation-func layer)))
    (declare (optimize (speed 3) (safety 0))
             (type (simple-array single-float) u-vec z-vec)
             (type (function (single-float) single-float) activation-func))
    (loop for i fixnum from 0 below (length z-vec) do
      (setf (aref z-vec i)
            (funcall activation-func (aref u-vec i))))
    z-vec))

(defun forward (in-vec nn)
  (loop for i from 0 to (1- (nn-n-layers nn)) do
    (if (zerop i)
        (progn (calc-u-vec in-vec (aref (nn-layer-vec nn) i))
               (calc-z-vec (aref (nn-layer-vec nn) i)))
        (progn (calc-u-vec (layer-z-vec (aref (nn-layer-vec nn) (1- i))) (aref (nn-layer-vec nn) i))
               (calc-z-vec (aref (nn-layer-vec nn) i))))))

;;; Back-propagation

(defun backward (train-vec nn)
  (declare (optimize (speed 3) (safety 0)))

  (let ((layer-vec (nn-layer-vec nn))
        (n-layers (nn-n-layers nn)))
    (declare (type (simple-array layer) layer-vec)
             (type fixnum n-layers))

    ;; calculate last layer's delta
    (let* ((last-layer (aref layer-vec (1- n-layers)))
           (z-vec (layer-z-vec last-layer))
           (delta-vec (layer-delta-vec last-layer)))
      (declare (type (simple-array single-float) train-vec z-vec delta-vec))
      (loop for j fixnum from 0 below (layer-out-dim last-layer) do
        (setf (aref delta-vec j) (- (aref z-vec j)
                                    (aref train-vec j)))))

    ;; calculate other deltas
    (loop for l fixnum from (- n-layers 2) downto 0 do
      (let ((layer (aref layer-vec l))
            (next-layer (aref layer-vec (1+ l))))
        (let ((u-vec (layer-u-vec layer))
              (delta-vec (layer-delta-vec layer))
              (next-delta-vec (layer-delta-vec next-layer))
              (next-w-mat (layer-w-mat next-layer))
              (activation-func-diff (layer-activation-func-diff layer)))
          (declare (type (simple-array single-float) next-delta-vec next-w-mat u-vec delta-vec)
                   (type (function (single-float) single-float) activation-func-diff))
          (loop for j fixnum from 0 below (layer-in-dim next-layer) do
            (setf (aref delta-vec j)
                  (* (funcall activation-func-diff (aref u-vec j))
                     (loop for k fixnum from 0 below (layer-out-dim next-layer)
                           summing
                           (* (aref next-delta-vec k) (aref next-w-mat k j))
                           single-float)))))))))

(defun update (in-vec train-vec nn)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array single-float) in-vec train-vec))
  
  (forward in-vec nn)
  (backward train-vec nn)

  (let ((layer-vec (nn-layer-vec nn))
        (n-layers (nn-n-layers nn))
        (learning-rate (nn-learning-rate nn)))
    (declare (type (simple-array layer) layer-vec)
             (type fixnum n-layers)
             (type single-float learning-rate))

    ;; update first layer
    (let* ((first-layer (aref layer-vec 0))
           (w-mat (layer-w-mat first-layer))
           (delta-vec (layer-delta-vec first-layer)))
      (declare (type (simple-array single-float) w-mat delta-vec))
      (loop for i fixnum from 0 below (layer-in-dim first-layer) do
        (loop for j fixnum from 0 below (layer-out-dim first-layer) do
          (setf (aref w-mat j i) (- (aref w-mat j i)
                                    (* learning-rate
                                       (aref in-vec i)
                                       (aref delta-vec j)))))))
    ;; update other layer
    (loop for l fixnum from 1 below n-layers do
      (let* ((layer (aref layer-vec l))
             (w-mat (layer-w-mat layer))
             (delta-vec (layer-delta-vec layer))
             (prev-layer (aref layer-vec (1- l)))
             (prev-z-vec (layer-z-vec prev-layer)))
        (declare (type (simple-array single-float) w-mat delta-vec prev-z-vec))
        (loop for i fixnum from 0 below (layer-in-dim layer) do
          (loop for j fixnum from 0 below (layer-out-dim layer) do
            (setf (aref w-mat j i) (- (aref w-mat j i)
                                      (* learning-rate
                                         (aref prev-z-vec i)
                                         (aref delta-vec j))))))))))

;;; predict

(defun predict (in-vec nn)
  (forward in-vec nn)
  (layer-z-vec (aref (nn-layer-vec nn) (1- (nn-n-layers nn)))))
