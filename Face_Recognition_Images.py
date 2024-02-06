import face_recognition as fr
import cv2


# cargar imagenes
foto_control = fr.load_image_file('FotoA.jpg')
foto_prueba = fr.load_image_file('FotoC.jpg')

# pasar imagenes a RGB
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_RGB2BGR)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_RGB2BGR)

# localizar rostros
lugar_rostro_A = fr.face_locations(foto_control)[0]
rostro_codificado_A = fr.face_encodings(foto_control)[0]

lugar_rostro_B = fr.face_locations(foto_prueba)[0]
rostro_codificado_B = fr.face_encodings(foto_prueba)[0]

# mostrar rectangulo
cv2.rectangle(foto_control,
              (lugar_rostro_A[3], lugar_rostro_A[0]),
              (lugar_rostro_A[1], lugar_rostro_A[2]),
              (0, 255, 0),
              5)

cv2.rectangle(foto_prueba,
              (lugar_rostro_B[3], lugar_rostro_B[0]),
              (lugar_rostro_B[1], lugar_rostro_B[2]),
              (0, 255, 0),
              5)

# comparar imagenes
resultado = fr.compare_faces([rostro_codificado_A], rostro_codificado_B)


if resultado[0]:
    color = (0, 255, 0)
    coicidencia = 'MATCH FOUND'

else:
    color = (255, 0, 0)
    coicidencia = 'MATCH NOT FOUND'



# medida de la distancia
distancia = fr.face_distance([rostro_codificado_A], rostro_codificado_B)
print(distancia)

# mostrar resultado 
cv2.putText(foto_prueba,
            f'{coicidencia} {distancia.round(2)}',
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            color,
            2)

# mostrar imagenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto Prueba', foto_prueba)

# mantener el programa abierto
cv2.waitKey(0)